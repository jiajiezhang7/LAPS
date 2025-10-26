import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from accelerate import Accelerator
import time
from torch.utils.tensorboard import SummaryWriter
import math
import random
import numpy as np
import os

from amplify_motion_tokenizer.dataset.velocity_dataset import get_dataloaders
from amplify_motion_tokenizer.models.motion_tokenizer import MotionTokenizer


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = 'none') -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha.clone().detach())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        if self.alpha is not None:
            alpha_factor = self.alpha.gather(0, targets)
        else:
            alpha_factor = 1.0

        focal_weight = (1.0 - target_probs).clamp(min=0.0).pow(self.gamma)
        loss = -alpha_factor * focal_weight * target_log_probs

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def train(config):
    # 1) Accelerator 与检查点目录
    accel_mprecision = str(config['training'].get('mixed_precision', 'no'))
    grad_accum_steps = int(config['training'].get('gradient_accumulation_steps', 1))
    accelerator = Accelerator(mixed_precision=accel_mprecision, gradient_accumulation_steps=grad_accum_steps)
    device = accelerator.device

    # 全局随机种子（可选，保证可复现与更稳定的码本利用）
    seed = config['training'].get('seed', None)
    try:
        seed_int = int(seed) if seed is not None else None
    except Exception:
        seed_int = None
    if seed_int is not None:
        random.seed(seed_int)
        np.random.seed(seed_int)
        torch.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            # 若部分算子不支持确定性，实现将回退为警告
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    
    # 创建专属于本次训练的子目录
    base_checkpoint_dir = Path(config['training']['checkpoint_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = base_checkpoint_dir / f"train_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.print(f"[Train] Created training-specific checkpoint directory: {checkpoint_dir}")

    # TensorBoard 日志目录与写入器（仅主进程）
    # 始终使用训练专属子目录下的tensorboard文件夹，忽略配置文件中的log_dir设置
    tb_log_dir = checkpoint_dir / "tensorboard"
    log_interval = int(config['training'].get('log_interval', 10))
    writer = None
    if accelerator.is_main_process:
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        accelerator.print(f"[Train] TensorBoard logs will be saved to: {tb_log_dir}")

    # 2) 数据
    accelerator.print("[Train] Loading data...")
    train_loader, val_loader = get_dataloaders(config)

    # 3) 模型
    accelerator.print("[Train] Initializing model...")
    model = MotionTokenizer(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"[Train] Model params: {n_params:,}")

    # 可选：从既有 checkpoint 恢复权重（受 resume 显式开关控制）
    resume_from_cfg = config['training'].get('resume_from', None)
    resume_flag_cfg = config['training'].get('resume', None)  # None 表示按是否配置了 resume_from 推断
    resume_enabled = bool(resume_flag_cfg) if (resume_flag_cfg is not None) else bool(resume_from_cfg)
    resume_from = resume_from_cfg if resume_enabled else None
    resume_optimizer_flag = bool(config['training'].get('resume_optimizer', False)) and resume_enabled
    if resume_from:
        ckpt_path = Path(resume_from)
        if ckpt_path.is_file():
            state = torch.load(str(ckpt_path), map_location='cpu')
            missing, unexpected = model.load_state_dict(state, strict=False)
            accelerator.print(f"[Train] Resumed weights from {ckpt_path}. missing={len(missing)} unexpected={len(unexpected)}")
        else:
            accelerator.print(f"[Train][Warn] resume_from path not found: {ckpt_path}")

    # 码本配置与日志开关
    fsq_levels = list(config['model'].get('fsq_levels', []))
    codebook_size_total = 1
    for L in fsq_levels:
        try:
            codebook_size_total *= int(L)
        except Exception:
            pass
    code_log_interval = int(config['training'].get('code_log_interval', config['training'].get('log_interval', 10)))
    code_entropy_weight = float(config['training'].get('code_entropy_weight', 0.0))
    code_entropy_warmup_epochs = int(config['training'].get('code_entropy_warmup_epochs', 0))
    warmup_offset_epochs = int(config['training'].get('code_entropy_warmup_offset', 0))

    # 工具函数：将 FSQ 多进制 digits -> 单整数 ID（同 inference 中实现的无注解版本）
    def _fsq_digits_to_ids(fsq_idx, levels):
        if fsq_idx is None:
            return None
        x = fsq_idx.detach().cpu().to(torch.long)
        # 去 batch 维
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.dim() == 2 and x.size(0) == 1:
            x = x.squeeze(0)
        # (d,) 直接返回
        if x.dim() == 1:
            return x
        # (d, k) 或 (k, d)
        if x.dim() == 2:
            d0, d1 = x.shape
            if len(levels) > 0 and d1 == len(levels):
                digits = x  # (d, k)
            elif len(levels) > 0 and d0 == len(levels):
                digits = x.transpose(0, 1)  # (d, k)
            else:
                # 无法判定 digits 维度，回退为展平的单维（非严格 code id，仅用于监控）
                return x.reshape(-1)
            # 预计算混合进制权重（小端）
            weights = torch.ones((len(levels),), dtype=torch.long)
            cur = 1
            for j, L in enumerate(levels):
                weights[j] = cur
                cur *= int(L)
            ids = (digits * weights.view(1, -1)).sum(dim=-1)  # (d,)
            return ids.to(torch.long)
        # 其他形状：回退为展平
        return x.reshape(-1)

    # 记录环境/超参信息
    world_size = accelerator.num_processes
    per_device_batch = int(config['training']['batch_size'])
    grad_accum = getattr(accelerator, 'gradient_accumulation_steps', 1)
    eff_global_batch = per_device_batch * world_size * grad_accum
    if writer is not None:
        writer.add_text("setup/world", f"world_size={world_size}")
        writer.add_text("setup/batch", f"per_device_batch={per_device_batch}, grad_accum={grad_accum}, effective_global_batch={eff_global_batch}")
        try:
            # 记录配置，便于复现实验
            writer.add_text("setup/config_yaml", yaml.dump(config))
        except Exception:
            pass

    # 4) 优化器与损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))

    focal_gamma = float(config['training'].get('focal_gamma', 2.0))
    focal_alpha_center = float(config['training'].get('focal_alpha_center', 0.25))
    focal_alpha_other = float(config['training'].get('focal_alpha_other', 1.0))
    num_classes = int(config['model']['num_classes'])
    decoder_window_size = int(config['model'].get('decoder_window_size', int(math.sqrt(num_classes))))
    center_class_index = (decoder_window_size // 2) * decoder_window_size + (decoder_window_size // 2)
    alpha_tensor = torch.full((num_classes,), focal_alpha_other, dtype=torch.float32)
    if 0 <= center_class_index < num_classes:
        alpha_tensor[center_class_index] = focal_alpha_center
    focal_loss_fn = FocalLoss(gamma=focal_gamma, alpha=alpha_tensor, reduction='none').to(device)

    # 若需要，尽早恢复优化器状态（需保存过对应文件）
    if resume_from and resume_optimizer_flag:
        # 推断优化器状态文件名（按 best/epoch_xx/last 三类）
        ckpt_p = Path(resume_from)
        opt_path = None
        if ckpt_p.name == 'best.pth':
            opt_path = ckpt_p.with_name('best_optim.pth')
        elif ckpt_p.name.startswith('motion_tokenizer_epoch_') and ckpt_p.suffix == '.pth':
            stem = ckpt_p.stem.replace('motion_tokenizer_epoch_', '')
            opt_path = ckpt_p.with_name(f"optimizer_epoch_{stem}.pth")
        elif ckpt_p.name == 'last.pth':
            opt_path = ckpt_p.with_name('last_optim.pth')
        if opt_path and opt_path.exists():
            try:
                optimizer.load_state_dict(torch.load(str(opt_path), map_location='cpu'))
                accelerator.print(f"[Train] Resumed optimizer state from {opt_path}")
            except Exception as e:
                accelerator.print(f"[Train][Warn] failed to load optimizer state: {e}")

    # 使用 Accelerate 准备分布式/多 GPU 训练
    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # 5) 训练循环
    epochs = int(config['training']['epochs'])
    accelerator.print("[Train] Starting training...")
    global_step = 0
    best_metric = float('inf')
    best_ckpt_path = checkpoint_dir / "best.pth"
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_local_samples = 0
        epoch_local_correct = 0
        epoch_local_total = 0
        epoch_start = time.time()
        # 本 epoch 内观测到的 code id（仅本进程用于日志）
        epoch_code_ids_set = set()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_local_main_process)
        for batch in pbar:
            # 兼容两种批数据格式：
            # 1) (velocities, labels, mask)  - 新增静态点掩码
            # 2) (velocities, labels)        - 旧格式（默认全部参与）
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                velocities, labels, mask = batch
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                velocities, labels = batch
                mask = torch.ones_like(labels, dtype=torch.bool)
            else:
                raise ValueError("Unexpected batch format; expected 2-tuple or 3-tuple")
            # 显式放置到当前设备
            velocities = velocities.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            with accelerator.accumulate(model):
                logits = model(velocities)  # (B, T-1, N, C)
                logits_reshaped = logits.reshape(-1, int(config['model']['num_classes']))
                labels_reshaped = labels.reshape(-1)
                mask_flat = mask.reshape(-1)

                # 元素级 CE，并按掩码加权平均（保持为主损失，向后兼容）
                loss_vec = focal_loss_fn(logits_reshaped, labels_reshaped)
                ce_loss = (loss_vec * mask_flat.float()).sum() / mask_flat.float().sum().clamp_min(1.0)

                # 阶段1：码本熵正则（可选，默认权重为 0，保持兼容）
                # 说明：此实现基于离散 code 直方图的熵，梯度对索引为零，仅作启发式正则。
                # 若需可导近似，需要额外设计（后续阶段再引入）。
                lambda_target = float(code_entropy_weight)
                if code_entropy_warmup_epochs > 0:
                    warmup_factor = min(1.0, float(epoch + 1) / float(max(1, code_entropy_warmup_epochs)))
                else:
                    warmup_factor = 1.0
                lambda_now = lambda_target * warmup_factor

                ent_norm_t = None
                if lambda_now > 0.0:
                    try:
                        # 优先从当前 wrapped 模型读取（一般可直连属性）
                        fsq_raw_current = getattr(model, 'last_fsq_indices', None)
                        if fsq_raw_current is None:
                            # 回退到解包后的模型（主/从进程均可读属性，但不做聚合）
                            fsq_raw_current = getattr(accelerator.unwrap_model(model), 'last_fsq_indices', None)
                        if isinstance(fsq_raw_current, torch.Tensor) and fsq_raw_current.numel() > 0:
                            code_ids = _fsq_digits_to_ids(fsq_raw_current, fsq_levels)
                            if code_ids is not None and code_ids.numel() > 0:
                                code_ids = code_ids.reshape(-1).to(torch.long)
                                eff_size = int(code_ids.max().item()) + 1
                                denom_K = int(codebook_size_total) if codebook_size_total > 1 else eff_size
                                counts = torch.bincount(code_ids, minlength=max(1, eff_size)).float()
                                total_count = counts.sum().clamp_min(1.0)
                                p = counts / total_count
                                nz = p > 0
                                ent = -(p[nz] * p[nz].log()).sum()  # nats, torch scalar
                                ent_norm_t = ent / math.log(max(2.0, float(denom_K)))
                    except Exception:
                        ent_norm_t = None

                if ent_norm_t is not None and lambda_now > 0.0:
                    loss_total = ce_loss + (- float(lambda_now)) * ent_norm_t
                else:
                    loss_total = ce_loss

                accelerator.backward(loss_total)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += ce_loss.item()
            epoch_local_samples += velocities.size(0)

            # 计算本步准确率（仅在掩码为 True 的位置）
            with torch.no_grad():
                preds = logits_reshaped.argmax(dim=1)
                correct = ((preds == labels_reshaped) & mask_flat).sum()
                total = mask_flat.sum()
                step_acc = (correct.float() / total).item()
                epoch_local_correct += int(correct.item())
                epoch_local_total += int(total)
            # 显示 CE 与 Total Loss（避免引用未定义变量）
            try:
                pbar.set_postfix(ce=f"{ce_loss.item():.4f}", total=f"{loss_total.item():.4f}")
            except Exception:
                pbar.set_postfix(ce=f"{ce_loss.item():.4f}")

            # 每隔 log_interval 步记录一次 step 级指标（仅主进程）
            if writer is not None and (global_step % log_interval == 0):
                # CE 与 Total Loss
                writer.add_scalar("train/loss_step", ce_loss.item(), global_step)
                try:
                    writer.add_scalar("train/loss_total_step", float(loss_total.item()), global_step)
                    writer.add_scalar("train/ce_loss_step", ce_loss.item(), global_step)
                except Exception:
                    pass
                # 记录当前学习率（取第一个 param_group）
                try:
                    lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar("train/lr", lr, global_step)
                except Exception:
                    pass
                writer.add_scalar("train/acc_step", step_acc, global_step)
                # 动态点比例（阶段0）
                try:
                    dyn_ratio_step = float(mask_flat.float().mean().item())
                    writer.add_scalar("train/dynamic_point_ratio_step", dyn_ratio_step, global_step)
                except Exception:
                    pass

            # 记录码本使用统计（每 code_log_interval 步，且仅主进程）
            if writer is not None and (global_step % code_log_interval == 0):
                try:
                    with torch.no_grad():
                        unwrapped = accelerator.unwrap_model(model)
                        fsq_raw = getattr(unwrapped, 'last_fsq_indices', None)
                        if isinstance(fsq_raw, torch.Tensor) and fsq_raw.numel() > 0:
                            # 将 digits 转为单一 code id（若可行），否则展平使用
                            code_ids = _fsq_digits_to_ids(fsq_raw, fsq_levels)
                            if code_ids is None or code_ids.numel() == 0:
                                raise RuntimeError("Empty code_ids after conversion")
                            code_ids = code_ids.reshape(-1).to(torch.long)

                            # 更新 epoch 级唯一码集合（本进程）
                            # 注意：该集合仅用于本地/主进程统计，不做多进程聚合
                            if 'epoch_code_ids_set' not in locals():
                                epoch_code_ids_set = set()
                            epoch_code_ids_set.update(int(v) for v in code_ids.tolist())

                            # 唯一码数量（合并后的 code id）
                            uniq = int(torch.unique(code_ids).numel())
                            writer.add_scalar("train/unique_codes_step", uniq, global_step)

                            # 唯一码数量（直接从原始 fsq_indices 计算）
                            try:
                                raw_unique = int(torch.unique(fsq_raw.reshape(-1)).numel())
                                writer.add_scalar("train/unique_codes_raw_step", raw_unique, global_step)
                            except Exception:
                                pass

                            # 频次分布 + perplexity/entropy
                            eff_size = int(code_ids.max().item()) + 1
                            counts = torch.bincount(code_ids, minlength=max(1, eff_size)).float()
                            total_count = counts.sum().clamp_min(1.0)
                            p = counts / total_count
                            # 避免 log(0)
                            nonzero = p > 0
                            ent = float((-p[nonzero] * p[nonzero].log()).sum().item())  # nats
                            ppl = float(math.exp(ent))
                            # 归一化熵（相对配置的码本大小；若未配置则按 eff_size）

                            # 仅作为记录：CE 主损失 + 熵正则的加权组合（不用于反传）
                            total_loss_for_log = float(ce_loss.item()) + float(-code_entropy_weight * ent_norm)
                            writer.add_scalar("train/total_loss_step", total_loss_for_log, global_step)
                        else:
                            # 若量化器未返回索引，仅记录占位
                            writer.add_scalar("train/unique_codes_step", 0, global_step)
                except Exception as e:
                    # 避免训练中断，仅打印一次性警告到控制台主进程
                    accelerator.print(f"[Warn][CodeLog] step={global_step}: {e}")
            global_step += 1

        # 计算全局平均损失
        local_avg = torch.tensor(total_loss / max(1, len(train_loader)), device=device)
        avg_loss = accelerator.reduce(local_avg, reduction="mean").item()
        # 样本数（全局）与 epoch 用时
        global_samples = accelerator.reduce(torch.tensor(epoch_local_samples, device=device, dtype=torch.float32), reduction="sum").item()
        # 计算全局准确率
        global_correct = accelerator.reduce(torch.tensor(epoch_local_correct, device=device, dtype=torch.float32), reduction="sum").item()
        global_total = accelerator.reduce(torch.tensor(epoch_local_total, device=device, dtype=torch.float32), reduction="sum").item()
        epoch_acc = (global_correct / max(1.0, global_total)) if global_total > 0 else 0.0
        epoch_time = time.time() - epoch_start
        samples_per_sec = (global_samples / epoch_time) if epoch_time > 0 else 0.0
        accelerator.print(f"[Train] Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Acc: {epoch_acc:.4f} | Samples: {int(global_samples)} | Time: {epoch_time:.2f}s | {samples_per_sec:.1f} samples/s")

        # 记录到 TensorBoard（仅主进程）
        if writer is not None:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)
            writer.add_scalar("train/acc_epoch", epoch_acc, epoch)
            writer.add_scalar("train/samples_per_epoch", global_samples, epoch)
            writer.add_scalar("train/epoch_time_sec", epoch_time, epoch)
            writer.add_scalar("train/samples_per_sec", samples_per_sec, epoch)
            # 该 epoch 内（本进程）观测到的唯一码数量（若有采样）
            try:
                if 'epoch_code_ids_set' in locals():
                    writer.add_scalar("train/unique_codes_epoch_local", float(len(epoch_code_ids_set)), epoch)
                    # 清空集合以免跨 epoch 累计
                    epoch_code_ids_set.clear()
            except Exception:
                pass
            writer.flush()

        # 验证评估
        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_correct_local = 0
            val_total_local = 0
            # 阶段0：验证集动态点占比统计
            val_dyn_true_local = 0
            val_dyn_all_local = 0
            with torch.no_grad():
                vbar = tqdm(val_loader, desc=f"Val {epoch+1}/{epochs}", disable=not accelerator.is_local_main_process)
                for v_batch in vbar:
                    if isinstance(v_batch, (tuple, list)) and len(v_batch) == 3:
                        v_vel, v_lbl, v_mask = v_batch
                    elif isinstance(v_batch, (tuple, list)) and len(v_batch) == 2:
                        v_vel, v_lbl = v_batch
                        v_mask = torch.ones_like(v_lbl, dtype=torch.bool)
                    else:
                        raise ValueError("Unexpected val batch format; expected 2-tuple or 3-tuple")
                    v_vel = v_vel.to(device)
                    v_lbl = v_lbl.to(device)
                    v_mask = v_mask.to(device)
                    v_logits = model(v_vel)
                    v_logits_r = v_logits.reshape(-1, int(config['model']['num_classes']))
                    v_lbl_r = v_lbl.reshape(-1)
                    v_mask_r = v_mask.reshape(-1)
                    v_loss_vec = focal_loss_fn(v_logits_r, v_lbl_r)
                    v_loss = (v_loss_vec * v_mask_r.float()).sum() / v_mask_r.float().sum().clamp_min(1.0)
                    val_total_loss += v_loss.item()
                    preds = v_logits_r.argmax(dim=1)
                    val_correct_local += int(((preds == v_lbl_r) & v_mask_r).sum().item())
                    val_total_local += int(v_mask_r.sum().item())
                    # 阶段0：累计掩码占比
                    val_dyn_true_local += int(v_mask_r.sum().item())
                    val_dyn_all_local += int(v_mask_r.numel())

            # 聚合 val 指标（所有进程都必须参与 reduce，避免分布式死锁）
            val_local_avg = torch.tensor(val_total_loss / max(1, len(val_loader)), device=device)
            val_loss = accelerator.reduce(val_local_avg, reduction="mean").item()
            val_correct = accelerator.reduce(torch.tensor(val_correct_local, device=device, dtype=torch.float32), reduction="sum").item()
            val_total = accelerator.reduce(torch.tensor(val_total_local, device=device, dtype=torch.float32), reduction="sum").item()
            # 阶段0：验证集动态点占比（全进程先聚合，再在主进程写日志）
            val_dyn_true_global = accelerator.reduce(torch.tensor(val_dyn_true_local, device=device, dtype=torch.float32), reduction="sum").item()
            val_dyn_all_global = accelerator.reduce(torch.tensor(val_dyn_all_local, device=device, dtype=torch.float32), reduction="sum").item()
            val_acc = (val_correct / max(1.0, val_total)) if val_total > 0 else 0.0
            accelerator.print(f"[Val] Epoch {epoch+1} | Avg Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

            if writer is not None:
                writer.add_scalar("val/loss_epoch", val_loss, epoch)
                writer.add_scalar("val/acc_epoch", val_acc, epoch)
                # 阶段0：验证集动态点占比（按 epoch 聚合）
                try:
                    val_dyn_ratio = float(val_dyn_true_global / max(1.0, val_dyn_all_global)) if val_dyn_all_global > 0 else 0.0
                    writer.add_scalar("val/dynamic_point_ratio_epoch", val_dyn_ratio, epoch)
                except Exception:
                    pass
                writer.flush()

        # 保存 best checkpoint（按验证损失优先；否则按训练损失）
        metric_to_track = val_loss if (val_loss is not None) else avg_loss
        if metric_to_track < best_metric:
            best_metric = metric_to_track
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), best_ckpt_path)
                # 额外保存优化器状态，便于后续完全续训
                try:
                    torch.save(optimizer.state_dict(), best_ckpt_path.with_name('best_optim.pth'))
                except Exception:
                    pass
                accelerator.print(f"[Train] Saved BEST checkpoint -> {best_ckpt_path} | metric={best_metric:.6f}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = checkpoint_dir / f"motion_tokenizer_epoch_{epoch+1}.pth"
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), ckpt_path)
                try:
                    torch.save(optimizer.state_dict(), ckpt_path.with_name(f"optimizer_epoch_{epoch+1}.pth"))
                except Exception:
                    pass
                accelerator.print(f"[Train] Saved checkpoint -> {ckpt_path}")

    # 保存最后一个 epoch 的权重为 last.pth
    last_ckpt_path = checkpoint_dir / "last.pth"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), last_ckpt_path)
        try:
            torch.save(optimizer.state_dict(), last_ckpt_path.with_name('last_optim.pth'))
        except Exception:
            pass
        accelerator.print(f"[Train] Saved LAST checkpoint -> {last_ckpt_path}")

    # 结束：写入 hparams 与关闭 TensorBoard（仅主进程）
    if writer is not None:
        try:
            # 以最后一个 epoch 的指标作为总结
            hparams = {
                "optimizer": str(config['training'].get('optimizer', 'AdamW')),
                "learning_rate": float(config['training'].get('learning_rate', 0.0001)),
                "batch_size": int(config['training'].get('batch_size', 0)),
                "epochs": int(config['training'].get('epochs', 0)),
                "num_layers": int(config['model'].get('num_layers', 0)),
                "num_heads": int(config['model'].get('num_heads', 0)),
                "hidden_dim": int(config['model'].get('hidden_dim', 0)),
                "decoder_window_size": int(config['model'].get('decoder_window_size', 0)),
                "num_classes": int(config['model'].get('num_classes', 0)),
                "fsq_levels": str(config['model'].get('fsq_levels', [])),
            }
            # 取最后记录的 epoch 指标（若未训练则置 0）
            final_epoch = max(0, epochs - 1)
            # 这里无法直接从 writer 读取，我们已经在循环内写入；再次写入作为 hparam metrics
            # 将最后一次计算的 avg_loss 和 epoch_acc 作为 final metrics
            writer.add_hparams(hparams, {
                "hparams/final_loss": float(avg_loss) if 'avg_loss' in locals() else 0.0,
                "hparams/final_acc": float(epoch_acc) if 'epoch_acc' in locals() else 0.0,
                "hparams/final_epoch": float(final_epoch),
            })
        except Exception:
            pass
        writer.flush()
        writer.close()


def override_config(config: dict, args: argparse.Namespace) -> dict:
    # 允许通过 CLI 覆盖常用训练参数
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.device is not None:
        config['training']['device'] = args.device
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers
    if args.log_dir is not None:
        config['training']['log_dir'] = args.log_dir
    if args.log_interval is not None:
        config['training']['log_interval'] = args.log_interval
    if args.mixed_precision is not None:
        config['training']['mixed_precision'] = args.mixed_precision
    if args.grad_accum_steps is not None:
        config['training']['gradient_accumulation_steps'] = args.grad_accum_steps
    # 新增：resume/seed 相关 CLI 覆盖
    if hasattr(args, 'seed') and (args.seed is not None):
        config['training']['seed'] = args.seed
    if hasattr(args, 'resume') and (args.resume is not None):
        config['training']['resume'] = bool(args.resume)
    if hasattr(args, 'resume_from') and (args.resume_from is not None):
        config['training']['resume_from'] = args.resume_from
    if hasattr(args, 'resume_optimizer') and (args.resume_optimizer is not None):
        config['training']['resume_optimizer'] = bool(args.resume_optimizer)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train the AMPLIFY Motion Tokenizer")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[0] / "configs" / "tokenizer_config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    # 新增：resume/seed 的 CLI 开关
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--resume-optimizer", dest="resume_optimizer", action="store_true")
    parser.add_argument("--no-resume-optimizer", dest="resume_optimizer", action="store_false")
    parser.set_defaults(resume_optimizer=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = override_config(config, args)
    train(config)


if __name__ == "__main__":
    main()
