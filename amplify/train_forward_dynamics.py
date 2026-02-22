import datetime
import math
import os
import sys

import hydra
import pyinstrument
import torch
from einops import rearrange
from IPython.core import ultratb
from omegaconf import OmegaConf
from torch.backends import opt_einsum
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import wandb
from amplify.models.encoders.t5 import T5
from amplify.models.encoders.vision_encoders import VisionEncoder
from amplify.models.forward_dynamics import ForwardDynamics
from amplify.models.motion_tokenizer import load_motion_tokenizer
from amplify.utils.cfg_utils import merge_checkpoint_config, get_device
from amplify.utils.data_utils import velocities_to_points
from amplify.utils.logger import Logger
from amplify.utils.metrics import get_traj_metrics
from amplify.utils.train import (
    batch_to_device,
    get_checkpoint_dir,
    get_dataloaders,
    get_datasets,
    get_root_dir,
    get_vis_dataset,
    index_batch,
    latest_checkpoint_from_dir,
    load_checkpoint,
    save_checkpoint,
)
from amplify.utils.vis_utils import vis_pred
from eval_libero import eval

torch.multiprocessing.set_start_method('spawn', force=True)
opt_einsum.strategy = 'auto-hq' # seems to speed up training by 10% or so


def train_epoch(train_global_iter, train_loader, models, optimizer, scaler, device, cfg, motion_tokenizer_cfg):
    traj_model = models["forward_dynamics"]
    traj_model.train()
    grad_accum = math.ceil(cfg.batch_size / cfg.gpu_max_bs) # number of gradient accumulations
    loss_sum = 0
    optimizer.zero_grad(set_to_none=True)
    optim_cfg = cfg.optim

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        batch = batch_to_device(batch, device)
        device_str = str(traj_model.device).split(':')[0]
        with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=cfg.optim.automatic_mixed_precision):
            # Obs
            img = batch['images']
            b, v, h, w, c = img.shape
            img = rearrange(img, 'b v h w c -> (b v) h w c')
            img_tokens = models['img_encoder'](img)
            img_tokens = rearrange(img_tokens, '(b v) t d -> b (v t) d', v=v)
            obs = {'image': img_tokens}

            # Goal
            if not cfg.forward_dynamics.text_encoder.use_preprocessed_embs:
                text_emb = models['text_encoder'](batch['text']).unsqueeze(1).to(device)
            else:
                text_emb = batch['text_emb']
            goal = {'text_emb': text_emb}

            # GT codes
            traj_gt = batch['traj'].to(device)
            traj_vel_gt = traj_gt[:, :, 1:] - traj_gt[:, :, :-1]

            if motion_tokenizer_cfg.cond_on_img:
                z = models["motion_tokenizer"].encode(traj_vel_gt, img)
            else:
                z = models["motion_tokenizer"].encode(traj_vel_gt)
            gt_codes, gt_indices = models["motion_tokenizer"].quantize(z)

            # Forward pass
            pred_indices, loss = traj_model(obs, goal, targets=gt_indices.long())

            pred_codes = models["motion_tokenizer"].quantize.indices_to_codes(pred_indices)

            # Decode tracks
            pred_traj_velocities, _ = models["motion_tokenizer"].decode(pred_codes)
            pred_traj = velocities_to_points(
                pred_traj_velocities, time_dim=2, init_points=batch["traj"][:, :, [0]]
            )

        # Backprop
        scaler.scale(loss).backward()
        if (batch_idx + 1) % grad_accum == 0:
            if optim_cfg.clip_grad > 0:
                scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(traj_model.parameters(), max_norm=optim_cfg.clip_grad) # norm is unaffected by unscaled gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_sum += loss.detach()

        # Logging
        traj_metrics = get_traj_metrics(pred_traj=pred_traj, gt_traj=batch['traj'], img_size=motion_tokenizer_cfg.img_shape)
        for key, value in traj_metrics.items():
            wandb.log({f"metrics/train_traj_{key}": value, "train_global_iter": train_global_iter})

        wandb.log({"train_loss": loss, "train_global_iter": train_global_iter})

        train_global_iter += 1

    loss_avg = loss_sum / len(train_loader)

    return loss_avg, train_global_iter


@torch.no_grad()
def val_epoch(val_global_iter, val_loader, models, device, cfg, motion_tokenizer_cfg):
    traj_model = models["forward_dynamics"]
    traj_model.eval()
    loss_sum = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
        batch = batch_to_device(batch, device)
        device_str = str(traj_model.device).split(':')[0]
        with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=cfg.optim.automatic_mixed_precision):
            # Obs
            img = batch['images']
            b, v, h, w, c = img.shape
            img = rearrange(img, 'b v h w c -> (b v) h w c')
            img_tokens = models['img_encoder'](img)
            img_tokens = rearrange(img_tokens, '(b v) t d -> b (v t) d', v=v)
            obs = {'image': img_tokens}

            # Goal
            if not cfg.forward_dynamics.text_encoder.use_preprocessed_embs:
                text_emb = models['text_encoder'](batch['text']).unsqueeze(1).to(device)
            else:
                text_emb = batch['text_emb']
            goal = {'text_emb': text_emb}

            # GT codes
            traj_gt = batch['traj'].to(device)
            traj_vel_gt = traj_gt[:, :, 1:] - traj_gt[:, :, :-1]

            if motion_tokenizer_cfg.cond_on_img:
                z = models["motion_tokenizer"].encode(traj_vel_gt, img)
            else:
                z = models["motion_tokenizer"].encode(traj_vel_gt)
            gt_codes, gt_indices = models["motion_tokenizer"].quantize(z)

            # Forward pass
            pred_indices, loss = traj_model(obs, goal, targets=gt_indices.long())

            pred_codes = models["motion_tokenizer"].quantize.indices_to_codes(pred_indices)

            # Decode tracks
            pred_traj_velocities, _ = models["motion_tokenizer"].decode(pred_codes)
            pred_traj = velocities_to_points(
                pred_traj_velocities, time_dim=2, init_points=batch["traj"][:, :, [0]]
            )

        loss_sum += loss.detach()

        # Logging
        traj_metrics = get_traj_metrics(pred_traj=pred_traj, gt_traj=batch['traj'], img_size=motion_tokenizer_cfg.img_shape)
        for key, value in traj_metrics.items():
            wandb.log({f"metrics/val_traj_{key}": value, "val_global_iter": val_global_iter})

        wandb.log({"val_loss": loss, "val_global_iter": val_global_iter})

        val_global_iter += 1

    loss_avg = loss_sum / len(val_loader)

    return loss_avg, val_global_iter


@torch.no_grad()
def generate_video(models, dataset, cfg):
    """
    Generates videos of gt and model predictions on sample rollout from a dataset
    """
    traj_model = models["forward_dynamics"]
    device_str = str(traj_model.device).split(':')[0]
    # Sample video from dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    video_idx = torch.randint(0, len(dataset), (1,)).item()
    full_batch = dataset.get_full_episode_batch(idx=video_idx)
    full_batch = batch_to_device(full_batch, device_str)
    vis_images = full_batch["images"]

    # GT video
    gt_traj = full_batch["traj"]
    gt_video = vis_pred(vis_images, gt_traj)
    gt_video = gt_video.permute(0, 3, 1, 2).cpu().numpy()

    # Pred video
    traj_model.eval()
    traj_len = full_batch["traj"].shape[0]
    with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=cfg.optim.automatic_mixed_precision):
        # Since the video may be too long, split into chunks of size
        # gpu_max_bs and then concatenate the results
        pred_trajs = []
        for start_t in tqdm(range(0, traj_len, cfg.gpu_max_bs)):
            end_t = min(start_t + cfg.gpu_max_bs, traj_len)
            indices = torch.arange(start_t, end_t)
            ibatch = index_batch(full_batch, indices)

            # Obs
            img = ibatch['images']
            b, v, h, w, c = img.shape
            img = rearrange(img, 'b v h w c -> (b v) h w c')
            img_tokens = models['img_encoder'](img)
            img_tokens = rearrange(img_tokens, '(b v) t d -> b (v t) d', v=v)
            obs = {'image': img_tokens}

            # Goal
            if not cfg.forward_dynamics.text_encoder.use_preprocessed_embs:
                text_emb = models['text_encoder'](ibatch['text']).unsqueeze(1).to(traj_model.device)
            else:
                text_emb = ibatch['text_emb']
            goal = {'text_emb': text_emb}

            # Predict
            pred_indices, _ = traj_model(obs, goal)
            pred_codes = models["motion_tokenizer"].quantize.indices_to_codes(pred_indices)

            # Decode tracks
            pred_traj_velocities, _ = models["motion_tokenizer"].decode(pred_codes)
            pred_traj = velocities_to_points(
                pred_traj_velocities, time_dim=2, init_points=ibatch["traj"][:, :, [0]]
            )

            pred_trajs.append(pred_traj)

        pred_trajs = torch.cat(pred_trajs, dim=0)
    pred_video = vis_pred(vis_images, pred_trajs)
    pred_video = pred_video.permute(0, 3, 1, 2).cpu().numpy()

    return gt_video, pred_video


@hydra.main(config_path="cfg", config_name='train_forward_dynamics', version_base='1.2')
def main(cfg):
    run_name = str(cfg.run_name) or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    assert cfg.forward_dynamics.motion_tokenizer.checkpoint is not None, "Track encoder checkpoint is required"
    if cfg.checkpoint is not None:
        cfg = merge_checkpoint_config(cfg)

    print("================== FINAL CONFIG ==================")
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    print("Using device: ", device)
    if cfg.profile:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        profiler = pyinstrument.Profiler()

    if cfg.root_dir is None:
        cfg.root_dir = get_root_dir()

    # --- MODELS ---
    models = {}
    # Load track encoder
    models["motion_tokenizer"], motion_tokenizer_cfg = load_motion_tokenizer(cfg.forward_dynamics.motion_tokenizer.checkpoint, frozen=cfg.forward_dynamics.motion_tokenizer.frozen)
    models["motion_tokenizer"] = models["motion_tokenizer"].to(device).eval()

    # Load vision encoder
    models["img_encoder"] = VisionEncoder(**cfg.forward_dynamics.vision_encoder)
    models["img_encoder"] = models["img_encoder"].to(device).eval()

    # Load text encoder
    if not cfg.forward_dynamics.text_encoder.use_preprocessed_embs:
        models["text_encoder"] = T5(**cfg.forward_dynamics.text_encoder)
        models["text_encoder"] = models["text_encoder"].to(device).eval()
        text_embed_dim = models["text_encoder"].embed_dim
        text_seq_len = models["text_encoder"].seq_len
    else:
        models["text_encoder"] = None
        text_embed_dim = 512
        text_seq_len = 1

    # Load forward dynamics model
    num_views = len(motion_tokenizer_cfg.cond_cameraviews)
    cond_seq_len = models["img_encoder"].seq_len * num_views + text_seq_len
    pred_seq_len = motion_tokenizer_cfg.track_pred_horizon - 1 # NOTE: assumes velocities

    models["forward_dynamics"] = ForwardDynamics(
        trunk_cfg=cfg.forward_dynamics.transformer,
        hidden_dim=motion_tokenizer_cfg.hidden_dim,
        img_dim=models["img_encoder"].embed_dim,
        text_dim=text_embed_dim,
        cond_seq_len=cond_seq_len,
        pred_seq_len=pred_seq_len,
        codebook_size=motion_tokenizer_cfg.codebook_size,
        quantize=models["motion_tokenizer"].quantize
    ).to(device)

    if cfg.compile:
        for model in models.values():
            model = torch.compile(model)

    # Dataloaders
    keys_to_load = motion_tokenizer_cfg.keys_to_load # tracks, images
    if cfg.forward_dynamics.text_encoder.use_preprocessed_embs:
        keys_to_load.append('text_emb')
    else:
        keys_to_load.append('text')
    keys_to_load = list(set(keys_to_load))

    train_datasets, val_datasets = get_datasets(
        root_dir=cfg.root_dir,
        train_datasets=cfg.train_datasets,
        val_datasets=cfg.val_datasets,
        keys_to_load=keys_to_load,
        motion_tokenizer_cfg=motion_tokenizer_cfg,
        aug_cfg=cfg.augmentations,
        task_names=cfg.task_names,
    )
    train_dataloader_dict, val_dataloader_dict = get_dataloaders(
        train_datasets,
        val_datasets,
        gpu_max_bs=cfg.gpu_max_bs,
        num_workers=cfg.num_workers,
        quick=cfg.quick
    )
    train_loader = train_dataloader_dict['traj']

    if val_dataloader_dict is not None:
        val_loader = val_dataloader_dict['traj']
    else:
        val_loader = None

    # Optimizer
    optim_cfg = cfg.optim
    optimizer = torch.optim.AdamW(models["forward_dynamics"].parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay, betas=optim_cfg.adam_betas)
    warmup_epochs = cfg.num_epochs // 10
    if optim_cfg.lr_schedule == 'cosine':
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01,total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs,eta_min=optim_cfg.lr / 100)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    elif optim_cfg.lr_schedule is None:
        scheduler = None
    else:
        raise ValueError(f"Invalid lr_schedule: {cfg.lr_schedule}")

    scaler = torch.amp.GradScaler(enabled=cfg.optim.automatic_mixed_precision)

    # Checkpoint dir
    resume = (cfg.checkpoint is not None or cfg.resume)
    checkpoint_dir = get_checkpoint_dir(stage="forward_dynamics", run_name=run_name, resume=resume)
    print(f"Checkpoint dir: {checkpoint_dir}")
    if cfg.resume and cfg.checkpoint is None:
        cfg.checkpoint = latest_checkpoint_from_dir(checkpoint_dir)

    # Load checkpoint
    if cfg.checkpoint is not None:
        models["forward_dynamics"], optimizer, scheduler, scaler, checkpoint_cfg, checkpoint_info = load_checkpoint(cfg.checkpoint, models["forward_dynamics"], optimizer, scheduler, scaler)

        # run info
        start_epoch = checkpoint_info['epoch'] + 1
        end_epoch = cfg.num_epochs + 1
        train_global_iter = checkpoint_info['train_global_iter']
        val_global_iter = checkpoint_info['val_global_iter']
        wandb_run_id = checkpoint_info['wandb_run_id']
    else:
        start_epoch = 1
        end_epoch = cfg.num_epochs + 1
        train_global_iter = 0
        val_global_iter = 0
        wandb_run_id = None

    # Logging
    logger = Logger(train_log_interval=cfg.log_interval, val_log_interval=cfg.log_interval)
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.wandb_init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        config=wandb_cfg,
        name=run_name,
        group=cfg.wandb_group,
        mode='disabled' if not cfg.use_wandb else 'online',
        id=wandb_run_id,
        resume="allow" if cfg.checkpoint is not None else None,
        allow_val_change=True,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb.config.update({"motion_tokenizer_cfg":  OmegaConf.to_container(motion_tokenizer_cfg)}, allow_val_change=True)
    wandb.config.update({"checkpoint_dir":  checkpoint_dir}, allow_val_change=True)

    if 'SLURM_JOBID' in os.environ:
        wandb.config.update({'slurm_job_id': os.environ['SLURM_JOBID']}, allow_val_change=True)

    # Train
    for epoch in range(start_epoch, end_epoch):
        if cfg.profile:
            profiler.start()

        # Train Epoch
        train_loss, train_global_iter = train_epoch(train_global_iter, train_loader, models, optimizer, scaler, device, cfg, motion_tokenizer_cfg)
        if cfg.generate_video:
            print("Generating train videos...")
            vis_dataset, fps = get_vis_dataset(train_datasets)
            gt_video, pred_video = generate_video(models, vis_dataset, cfg)
            wandb.log({"train_pred_video": wandb.Video(pred_video, fps=fps, format="mp4"), "epoch": epoch})
            wandb.log({"train_gt_video": wandb.Video(gt_video, fps=fps, format="mp4"), "epoch": epoch})

        # Val Epoch
        if val_loader is not None:
            val_loss, val_global_iter = val_epoch(val_global_iter, val_loader, models, device, cfg, motion_tokenizer_cfg)
            if cfg.generate_video:
                print("Generating validation videos...")
                vis_dataset, fps = get_vis_dataset(val_datasets)
                gt_video, pred_video = generate_video(models, vis_dataset, cfg)
                wandb.log({"val_pred_video": wandb.Video(pred_video, fps=fps, format="mp4"), "epoch": epoch})
                wandb.log({"val_gt_video": wandb.Video(gt_video, fps=fps, format="mp4"), "epoch": epoch})
        else:
            val_loss = 0
            val_global_iter = 0

        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optim_cfg.lr

        # Log
        print(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | LR: {current_lr}")
        wandb.log({'avg_train_loss': train_loss, 'avg_val_loss': val_loss, 'learning_rate': current_lr, 'epoch': epoch})

        # Save checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        save_checkpoint(latest_path, epoch, cfg, models["forward_dynamics"], optimizer, scaler, train_loss, val_loss, train_global_iter, val_global_iter, scheduler)

        if epoch % cfg.save_interval == 0 and epoch > 0:
            # datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # checkpoint_name = f"checkpoints/{run_name}_epoch_{epoch}_{datetime_str}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
            save_checkpoint(checkpoint_path, epoch, cfg, models["forward_dynamics"], optimizer, scaler, train_loss, val_loss, train_global_iter, val_global_iter, scheduler)

        if cfg.profile:
            profiler.stop()
            profiler.print()


if __name__ == "__main__":
    main()
