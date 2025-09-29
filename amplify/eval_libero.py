import json
import os
from typing import List, Optional

import gym
import hydra
import numpy as np
import torch
from einops import repeat
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
import cv2
from amplify.utils.cfg_utils import get_device
from amplify.utils.kp_utils.query_utils import grid_queries
from amplify.utils.libero_utils.env_utils import build_libero_env
from amplify.utils.vis_utils import vis_pred

from amplify import AMPLIFY

def rollout(
    env: gym.Env,
    policy: AMPLIFY,
    task_desc: str,
    task_emb: torch.Tensor,
    max_steps: int = 500,
    cond_cameraviews: List[str] = ['agentview'],
    num_tracks: int = 400,
    temporal_agg: bool = True,
    action_horizon: int = 16,
    action_dim: int = 7,
    vis_traj: bool = True,
    img_size: int = 128,
):
    device = get_device()
    n_envs = env.env_num
    text = [task_desc] * n_envs
    text_emb = task_emb.repeat(n_envs, 1).to(device)
    # Seed query points for visualization
    init_queries = grid_queries(views=1, n_tracks=num_tracks, device=device).standard()

    obs = env.reset()

    step = 0
    dones = np.zeros(n_envs, dtype=bool)
    done_steps = np.ones(n_envs, dtype=np.uint8) * max_steps
    height = img_size
    # Always render views side-by-side for consistent frame shape
    num_views = len(cond_cameraviews)
    width = height * num_views
    obs_list = np.zeros((n_envs, max_steps, height, width, 3), dtype=np.uint8)
    if temporal_agg:
        all_time_actions = torch.zeros((n_envs, max_steps, action_horizon, action_dim)).to(device)

    for step in tqdm(range(max_steps), desc="Rollout Steps"):
        # Construct a batch from observations
        images = []
        for view in cond_cameraviews:
            try:
                image = torch.tensor(obs[f'{view}_image'])
            except KeyError:
                image = torch.tensor(obs[f'robot0_{view}_image'])
            images.append(image)
        images = torch.stack(images, dim=1).float().to(device) / 255.0

        proprio = torch.tensor(np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]], axis=-1)).float().to(device)

        traj = repeat(init_queries, '1 n d -> n_envs v 1 n d', v=len(cond_cameraviews), n_envs=n_envs)

        with torch.no_grad():
            # Use unified AMPLIFY act() method
            actions_full = policy.act(
                images=images,
                proprio=proprio,
                text=text,
                text_emb=text_emb,
                ar_sampling='argmax'
            )

            # For trajectory visualization, use predict_traj() if needed
            pred_traj = None
            if vis_traj:
                # move policy.motion_tokenizer to device if on cpu
                if policy.motion_tokenizer.decoder.device != device:
                    policy.motion_tokenizer.decoder.to(device)

                pred_traj = policy.predict_traj(
                    images=images,
                    init_queries=traj,
                    text=text,
                    text_emb=text_emb,
                    ar_sampling='argmax'
                )

        if temporal_agg:
            all_time_actions[:, step, :, :] = actions_full

            start_t = max(0, step - action_horizon + 1)
            num_actions = step - start_t + 1
            time_indices = torch.arange(start_t, step + 1).to(device)

            idx_in_action_horizon = (step - time_indices).to(device)
            batch_indices = torch.arange(n_envs).unsqueeze(1).repeat(1, num_actions).to(device)
            time_indices = time_indices.unsqueeze(0).repeat(n_envs, 1)
            idx_in_action_horizon = idx_in_action_horizon.unsqueeze(0).repeat(n_envs, 1).to(device)

            actions_for_curr_step = all_time_actions[batch_indices, time_indices, idx_in_action_horizon, :]
            actions_nonzero = torch.any(actions_for_curr_step != 0, dim=2)

            k = 0.01
            exp_weights = np.exp(-k * np.arange(num_actions))
            exp_weights = torch.from_numpy(exp_weights).float().to(device)
            weights = exp_weights.unsqueeze(0) * actions_nonzero.float()
            weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
            normalized_weights = weights / weights_sum
            actions = (actions_for_curr_step * normalized_weights.unsqueeze(2)).sum(dim=1)
            actions = actions.cpu().numpy()
        else:
            actions = actions_full[:, 0].cpu().numpy()

        obs, reward, done, info = env.step(actions)

        for k in range(n_envs):
            dones[k] = dones[k] or done[k]
            if dones[k]:
                done_steps[k] = step
            if vis_traj and pred_traj is not None:
                vis_batch_local = {'images': images, 'traj': pred_traj}
                obs_list[k, step] = vis_pred(vis_batch_local['images'], vis_batch_local['traj'])[k].cpu().numpy()
            else:
                # Concatenate per-view images horizontally for consistent layout
                images_u8 = (images * 255.0).clamp(0, 255).to(torch.uint8)
                concat = torch.cat([images_u8[:, vi] for vi in range(num_views)], dim=-2)  # (B, H, V*W, C)
                obs_list[k, step] = concat[k].cpu().numpy()

        if all(dones):
            break

        step += 1

    return obs_list, dones, done_steps

def save_rollout_video(obs_list: np.ndarray, done_steps: np.ndarray, log_name: str, run_name: str, fps: int = 15):
    print('Saving videos...')
    # Only save first video
    video_len = int(done_steps[0])
    frames = obs_list[0][:video_len]  # (T, H, W, 3), uint8, RGB

    # Log to wandb
    wandb_tensor = frames.transpose(0, 3, 1, 2)  # (T, 3, H, W)
    wandb_video = wandb.Video(wandb_tensor, fps=fps, format="mp4")
    wandb.log({f'{log_name}': wandb_video}, commit=False)

    # Save to local mp4 under results/<run_name>
    out_dir = os.path.join('results', str(run_name))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{log_name}.mp4")
    if frames.shape[0] > 0:
        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        print(f"Saved video to {os.path.abspath(out_path)}")


def eval(
        policy: AMPLIFY,
        suites: List[str],
        img_size: int,
        n_envs: int,
        max_steps: int = 500,
        cond_cameraviews: List[str] = ['agentview'],
        num_tracks: int = 400,
        temporal_agg: bool = True,
        action_horizon: int = 16,
        action_dim: int = 7,
        vis_traj: bool = True,
        log_detailed_stats: bool = False,
        log_step_key: str = 'eval_step',
        log_step: int = 0,
        save_video: bool = False,
        libero_path: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
    rollout_info = {}
    for suite in suites:
        assert suite in ['libero_10', 'libero_90', 'libero_spatial', 'libero_goal', 'libero_object']
        num_tasks = 90 if suite == 'libero_90' else 10
        rollout_info[suite] = {}

        for task_no in tqdm(range(num_tasks)):
            rollout_info[suite][task_no] = {}
            # Load env
            env, task_desc, task_emb = build_libero_env(
                task_suite=suite,
                task_no=task_no,
                img_size=img_size,
                n_envs=n_envs,
                dataset_path=libero_path,
            )
            print(f'Evaluating task no {task_no} from {suite}...')
            print(f'Task description: {task_desc}')

            obs_list, dones, done_steps = rollout(
                env=env,
                policy=policy,
                task_desc=task_desc,
                task_emb=task_emb,
                max_steps=max_steps,
                cond_cameraviews=cond_cameraviews,
                num_tracks=num_tracks,
                temporal_agg=temporal_agg,
                action_horizon=action_horizon,
                action_dim=action_dim,
                vis_traj=vis_traj,
                img_size=img_size,
            )

            if save_video:
                save_rollout_video(obs_list, done_steps, f'{suite}_task_{task_no}_video', run_name=run_name or 'default')

            rollout_info[suite][task_no]['success_rate'] = np.mean(dones)
            print(f'Suite: {suite}, Task: {task_no}, Success rate: {rollout_info[suite][task_no]["success_rate"]}')
            if log_detailed_stats:
                wandb.log({f'eval/{suite}_task_{task_no}_success_rate': rollout_info[suite][task_no]['success_rate'], log_step_key: log_step})

            env.close()

        rollout_info[suite]['suite_success_rate'] = np.mean([info['success_rate'] for info in rollout_info[suite].values() if isinstance(info, dict) and 'success_rate' in info])
        print(f"Suite {suite} success rate: {rollout_info[suite]['suite_success_rate']:.3f}")
        wandb.log({f'eval/{suite}_success_rate': rollout_info[suite]['suite_success_rate'], log_step_key: log_step})

    # Calculate overall success rate across all suites
    suite_success_rates = [info['suite_success_rate'] for info in rollout_info.values() if 'suite_success_rate' in info]
    if suite_success_rates:
        rollout_info['overall_success_rate'] = np.mean(suite_success_rates)
        print(f"Overall success rate: {rollout_info['overall_success_rate']:.3f}")
        wandb.log({'eval/overall_success_rate': rollout_info['overall_success_rate'], log_step_key: log_step})

    # Save rollout_info as json
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_path = os.path.join(hydra_output_dir, 'rollout_info.json')
    with open(output_path, 'w') as f:
        json.dump(rollout_info, f, indent=4)

    return rollout_info




@hydra.main(config_path='cfg', config_name='eval_libero', version_base='1.2')
def main(cfg):
    amplify_ckpt = getattr(cfg, 'amplify_checkpoint', None)
    assert amplify_ckpt is not None, 'Please provide an AMPLIFY checkpoint (cfg.amplify_checkpoint).'

    # Logging
    if cfg.use_wandb:
        config_dict = OmegaConf.to_container(cfg)
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            config=config_dict,
        )
    else:
        wandb.init(mode='disabled')

    if 'SLURM_JOBID' in os.environ:
        wandb.config.update({'slurm_job_id': os.environ['SLURM_JOBID']}, allow_val_change=True)

    device = get_device()

    # Load unified policy
    policy = AMPLIFY.load(amplify_ckpt, device=device, compile=cfg.compile)

    # Use bundled configs for evaluation shapes
    mt_cfg = policy.motion_tokenizer_cfg
    id_cfg = policy.id_cfg

    # Set run name and wandb name
    run_name = cfg.run_name or os.path.basename(amplify_ckpt)
    wandb.run.name = run_name

    # Log configs to wandb
    wandb.config.update({
        'motion_tokenizer_cfg': OmegaConf.to_container(mt_cfg),
        'fd_cfg': OmegaConf.to_container(policy.fd_cfg),
        'id_cfg': OmegaConf.to_container(id_cfg),
    }, allow_val_change=True)

    # Determine if we should visualize trajectories
    vis_traj = id_cfg.cond_on_tracks and cfg.vis_traj

    # offload the motion tokenizer to cpu
    policy.motion_tokenizer.encoder.to('cpu')
    policy.motion_tokenizer.decoder.to('cpu')

    # Run evaluation
    eval(
        policy=policy,
        suites=cfg.dataset,
        img_size=mt_cfg.img_shape[0],
        n_envs=cfg.n_envs,
        max_steps=cfg.max_steps,
        cond_cameraviews=mt_cfg.cond_cameraviews,
        num_tracks=mt_cfg.num_tracks,
        temporal_agg=cfg.temporal_agg,
        action_horizon=mt_cfg.true_horizon,
        action_dim=id_cfg.action_dim,
        vis_traj=vis_traj,
        log_detailed_stats=True,
        save_video=cfg.save_rollout_video,
        libero_path=cfg.libero_path,
        run_name=run_name,
    )

if __name__=='__main__':
    main()
