import math
import os
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pyinstrument
import torch
from einops import rearrange
from IPython.core import ultratb
from omegaconf import OmegaConf
from torch.backends import opt_einsum
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import wandb
from amplify.models.inverse_dynamics import InverseDynamics
from amplify.models.encoders.t5 import T5
from amplify.models.encoders.vision_encoders import VisionEncoder
from amplify.models.forward_dynamics import ForwardDynamics
from amplify.models.motion_tokenizer import load_motion_tokenizer, load_vae_encoder
from amplify.utils.cfg_utils import merge_checkpoint_config, get_device
from amplify.utils.data_utils import velocities_to_points
from amplify.utils.logger import Logger
from amplify.utils.train import (
    get_checkpoint_dir,
    get_dataloaders,
    get_datasets,
    get_root_dir,
    latest_checkpoint_from_dir,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)
from amplify.utils.vis_utils import vis_pred

# sys.excepthook = ultratb.FormattedTB(mode='Plain', color_scheme='Linux', call_pdb=1)

torch.multiprocessing.set_start_method('spawn', force=True)
opt_einsum.strategy = 'auto-hq' # seems to speed up training by 10% or so


def train_epoch(train_global_iter, models, train_loader, optimizer, scaler, device, logger, cfg, motion_tokenizer_cfg):
    models["inverse_dynamics"].train()
    grad_accum = math.ceil(cfg.batch_size / cfg.gpu_max_bs) # number of gradient accumulations
    train_loss = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        # Target
        action_gt = batch['actions'].to(device)

        # autocast not supported on mps because fp16/bf16 do not have complete support in mps and cause numerical stability and NaNs
        with torch.autocast(device_type=str(device).split(':')[0], dtype=torch.float16, enabled=cfg.amp and device.type != "mps"):
            # Conditioning inputs
            input_dict = {}

            with torch.no_grad():
                if motion_tokenizer_cfg.cond_on_img or cfg.cond_on_img or cfg.forward_dynamics_checkpoint is not None:
                    img = batch['images'].to(device)
                    img = rearrange(img, 'b v h w c -> (b v) h w c')
                    img_tokens = models["img_encoder"](img)
                    img_tokens = rearrange(img_tokens, '(b v) t d -> b t (v d)', v=len(motion_tokenizer_cfg.cond_cameraviews))
                    input_dict['img_tokens'] = img_tokens

                if cfg.cond_on_text or cfg.forward_dynamics_checkpoint is not None:
                    if cfg.text_encoder.use_preprocessed_embs:
                        input_dict['text_tokens'] = batch['text_emb'].to(device)
                    else:
                        input_dict['text_tokens'] = models["text_encoder"](batch['text']).unsqueeze(1).to(device)

                if cfg.cond_on_proprio:
                    input_dict['proprioception'] = batch['proprioception'].to(device).unsqueeze(1)

                if cfg.cond_on_tracks:
                    # Codes from forward dynamics model
                    if cfg.forward_dynamics_checkpoint is not None:
                        fd_img_tokens = rearrange(img_tokens, 'b t (v d) -> b (v t) d', v=len(motion_tokenizer_cfg.cond_cameraviews))
                        obs = {'image': fd_img_tokens}
                        goal = {'text_emb': input_dict['text_tokens']}
                        pred_indices, _ = models["forward_dynamics"](obs, goal)
                        input_dict['codes'] = models["motion_tokenizer"].quantize.indices_to_codes(pred_indices)

                    # Codes from motion tokenizer
                    else:
                        traj_gt = batch['traj'].to(device)
                        traj_vel_gt = traj_gt[:, :, 1:] - traj_gt[:, :, :-1]

                        if motion_tokenizer_cfg.cond_on_img:
                            z = models["motion_tokenizer"].encode(traj_vel_gt, img)
                        else:
                            z = models["motion_tokenizer"].encode(traj_vel_gt)
                        input_dict['codes'], _ = models["motion_tokenizer"].quantize(z)

                    # vis recon for debugging
                    if cfg.vis_recon:
                        x_recon, _ = models["motion_tokenizer"].decode(input_dict['codes'])
                        x_recon = velocities_to_points(x_recon.cpu(), time_dim=2, init_points=batch['traj'][:, :, [0]])
                        vis_img_pred = vis_pred(batch['images'], x_recon).numpy()
                        vis_img_gt = vis_pred(batch['images'], batch['traj']).numpy()
                        vis_img = np.concatenate([vis_img_gt, vis_img_pred], axis=1)
                        for vis in vis_img:
                            plt.imshow(vis)
                            plt.show()

            # Prediction
            if models["inverse_dynamics"].requires_action_seq:
                action_pred = models["inverse_dynamics"](input_dict, action_gt)
            else:
                action_pred = models["inverse_dynamics"](input_dict)

            # Loss
            loss = models["inverse_dynamics"].loss_fn(action_pred, action_gt)

        # Backprop
        scaler.scale(loss).backward()
        if (batch_idx + 1) % grad_accum == 0:
            if cfg.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(models["inverse_dynamics"].parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Logging
        train_loss += loss.detach()
        logger.update({"train_loss": loss.item()}, train_global_iter, phase='train')

        train_global_iter += cfg.gpu_max_bs

    avg_train_loss = train_loss.item() / len(train_loader)

    return avg_train_loss, train_global_iter

@torch.no_grad()
def val_epoch(val_global_iter, models, val_loader, device, logger, cfg, motion_tokenizer_cfg):
    models["inverse_dynamics"].eval()
    val_loss = 0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
        # Target
        action_gt = batch['actions'].to(device)

        with torch.autocast(device_type=str(device).split(':')[0], dtype=torch.float16, enabled=cfg.amp and device.type != "mps"):
            # Conditioning inputs
            input_dict = {}

            with torch.no_grad():
                if motion_tokenizer_cfg.cond_on_img or cfg.cond_on_img or cfg.forward_dynamics_checkpoint is not None:
                    img = batch['images'].to(device)
                    img = rearrange(img, 'b v h w c -> (b v) h w c')
                    img_tokens = models["img_encoder"](img)
                    img_tokens = rearrange(img_tokens, '(b v) t d -> b t (v d)', v=len(motion_tokenizer_cfg.cond_cameraviews))
                    input_dict['img_tokens'] = img_tokens

                if cfg.cond_on_text or cfg.forward_dynamics_checkpoint is not None:
                    if cfg.text_encoder.use_preprocessed_embs:
                        input_dict['text_tokens'] = batch['text_emb'].to(device)
                    else:
                        input_dict['text_tokens'] = models["text_encoder"](batch['text']).unsqueeze(1).to(device)

                if cfg.cond_on_proprio:
                    input_dict['proprioception'] = batch['proprioception'].to(device).unsqueeze(1)

                if cfg.cond_on_tracks:
                    # Codes from forward dynamics model
                    if cfg.forward_dynamics_checkpoint is not None:
                        fd_img_tokens = rearrange(img_tokens, 'b t (v d) -> b (v t) d', v=len(motion_tokenizer_cfg.cond_cameraviews))
                        obs = {'image': fd_img_tokens}
                        goal = {'text_emb': input_dict['text_tokens']}
                        pred_indices, _ = models["forward_dynamics"](obs, goal)
                        input_dict['codes'] = models["motion_tokenizer"].quantize.indices_to_codes(pred_indices)

                    # Codes from motion tokenizer
                    else:
                        traj_gt = batch['traj'].to(device)
                        traj_vel_gt = traj_gt[:, :, 1:] - traj_gt[:, :, :-1]

                        if motion_tokenizer_cfg.cond_on_img:
                            z = models["motion_tokenizer"].encode(traj_vel_gt, img)
                        else:
                            z = models["motion_tokenizer"].encode(traj_vel_gt)
                        input_dict['codes'], _ = models["motion_tokenizer"].quantize(z)

            # Prediction
            if models["inverse_dynamics"].requires_action_seq:
                action_pred = models["inverse_dynamics"](input_dict, action_gt)
            else:
                action_pred = models["inverse_dynamics"](input_dict)

            # Loss
            loss = models["inverse_dynamics"].loss_fn(action_pred, action_gt)

        val_loss += loss.detach()

        # Logging
        logger.update({"val_loss": loss.item()}, val_global_iter, phase='val')

        val_global_iter += cfg.gpu_max_bs

    avg_val_loss = val_loss.item() / len(val_loader)

    return avg_val_loss, val_global_iter


@hydra.main(config_path="cfg", config_name="train_inverse_dynamics", version_base="1.2")
def main(cfg):
    seed_everything(cfg.seed)
    run_name = f"{cfg.run_name}_seed_{cfg.seed}"

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

    # Track encoder
    models["motion_tokenizer"], motion_tokenizer_cfg = load_motion_tokenizer(cfg.motion_tokenizer_checkpoint, frozen=True)
    models["motion_tokenizer"] = models["motion_tokenizer"].to(device).eval()

    # Vision encoder
    models["img_encoder"] = VisionEncoder(**cfg.vision_encoder).eval().to(device)
    cfg.num_img_tokens = models["img_encoder"].seq_len
    cfg.img_embed_dim = models["img_encoder"].embed_dim * len(motion_tokenizer_cfg.cond_cameraviews) # stack views in embedding dim, not seq dim

    # Text encoder
    if cfg.text_encoder.use_preprocessed_embs:
        cfg.text_embed_dim = 512
        text_seq_len = 1
    else:
        models["text_encoder"] = T5(**cfg.text_encoder).eval().to(device)
        cfg.text_embed_dim = models["text_encoder"].embed_dim
        text_seq_len = models["text_encoder"].seq_len


    models["inverse_dynamics"] = InverseDynamics(
        motion_tokenizer_cfg,
        cfg,
    ).to(device)

    if cfg.forward_dynamics_checkpoint is not None: # If using predicted codes
        # Load forward dynamics model
        num_views = len(motion_tokenizer_cfg.cond_cameraviews)
        cond_seq_len = models["img_encoder"].seq_len * num_views + text_seq_len
        pred_seq_len = motion_tokenizer_cfg.track_pred_horizon - 1 # -1 because it's velocity

        # Make less hacky
        fd_config = OmegaConf.create(torch.load(cfg.forward_dynamics_checkpoint, weights_only=False)['config'])

        models["forward_dynamics"] = ForwardDynamics(
            trunk_cfg=fd_config.forward_dynamics.transformer,
            hidden_dim=motion_tokenizer_cfg.hidden_dim,
            img_dim=models["img_encoder"].embed_dim,
            text_dim=cfg.text_embed_dim,
            cond_seq_len=cond_seq_len,
            pred_seq_len=pred_seq_len,
            codebook_size=motion_tokenizer_cfg.codebook_size,
            quantize=models["motion_tokenizer"].quantize
        ).to(device)

        # Load weights
        models["forward_dynamics"] = load_checkpoint(cfg.forward_dynamics_checkpoint, models["forward_dynamics"])[0]

    if cfg.compile:
        for model in models.values():
            model = torch.compile(model)

    # Dataloaders
    keys_to_load = motion_tokenizer_cfg.keys_to_load # tracks, images
    keys_to_load.append('actions')
    if cfg.cond_on_proprio:
        keys_to_load.append('proprioception')
    if cfg.text_encoder.use_preprocessed_embs:
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
        task_names=cfg.task_names,
        normalize_actions=cfg.normalize_actions,
        action_key=cfg.action_key,
    )
    train_dataloader_dict, val_dataloader_dict = get_dataloaders(
        train_datasets,
        val_datasets,
        gpu_max_bs=cfg.gpu_max_bs,
        num_workers=cfg.num_workers,
        quick=cfg.quick
    )
    train_loader = train_dataloader_dict['action']
    if val_dataloader_dict is not None:
        val_loader = val_dataloader_dict['action']
    else:
        val_loader = None

    # Optimizer, Scheduler
    optimizer = torch.optim.AdamW(models["inverse_dynamics"].parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.adam_betas)
    warmup_epochs = cfg.num_epochs // 10
    if cfg.lr_schedule == 'cosine':
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01,total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs,eta_min=cfg.lr / 100)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    elif cfg.lr_schedule is None:
        scheduler = None
    else:
        raise ValueError(f"Invalid lr_schedule: {cfg.lr_schedule}")
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type != "mps")

    # Checkpoint dir
    resume = (cfg.checkpoint is not None or cfg.resume)
    checkpoint_dir = get_checkpoint_dir(stage="inverse_dynamics", run_name=run_name, resume=resume)
    print(f"Checkpoint dir: {checkpoint_dir}")
    if cfg.resume and cfg.checkpoint is None:
        cfg.checkpoint = latest_checkpoint_from_dir(checkpoint_dir)

    # Load checkpoint
    if cfg.checkpoint is not None:
        models["inverse_dynamics"], optimizer, scheduler, scaler, checkpoint_cfg, checkpoint_info = load_checkpoint(cfg.checkpoint, models["inverse_dynamics"], optimizer, scheduler, scaler)

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
    )
    wandb.config.update({"motion_tokenizer_cfg":  OmegaConf.to_container(motion_tokenizer_cfg)}, allow_val_change=True)
    wandb.config.update({"checkpoint_dir":  checkpoint_dir}, allow_val_change=True)

    if 'SLURM_JOBID' in os.environ:
        wandb.config.update({'slurm_job_id': os.environ['SLURM_JOBID']}, allow_val_change=True)

    # Train
    for epoch in range(start_epoch, end_epoch):
        if cfg.profile:
            profiler.start()

        train_loss, train_global_iter = train_epoch(train_global_iter, models, train_loader, optimizer, scaler, device, logger, cfg, motion_tokenizer_cfg)

        if val_loader is not None:
            val_loss, val_global_iter = val_epoch(val_global_iter, models, val_loader, device, logger, cfg, motion_tokenizer_cfg)
        else:
            val_loss = 0
            val_global_iter = 0

        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = cfg.lr

        # Log
        print(f'Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | LR: {current_lr}')
        logger.log({"epoch": epoch, "learning_rate": current_lr, "avg_train_loss": train_loss, "avg_val_loss": val_loss}, train_global_iter) # "avg_train_loss": train_loss, "avg_val_loss": val_loss,

        # Save model
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        save_checkpoint(latest_path, epoch, cfg, models["inverse_dynamics"], optimizer, scaler, train_loss=train_loss, val_loss=val_loss, train_global_iter=train_global_iter, val_global_iter=val_global_iter, scheduler=scheduler)

        if epoch % cfg.save_interval == 0 and epoch > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
            save_checkpoint(checkpoint_path, epoch, cfg, models["inverse_dynamics"], optimizer, scaler, train_loss=train_loss, val_loss=val_loss, train_global_iter=train_global_iter, val_global_iter=val_global_iter, scheduler=scheduler)

        if cfg.profile:
            profiler.stop()
            profiler.print()



if __name__=='__main__':
    main()
