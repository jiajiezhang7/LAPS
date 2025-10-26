import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ
import inspect

from amplify_motion_tokenizer.models.components import KeypointEncoder, KeypointDecoder


class MotionTokenizer(nn.Module):
    """
    整合 Encoder, FSQ, Decoder 的自编码器模型。
    输入: velocities (B, T-1, N, 2)  ∈ [-1, 1]
    输出: logits (B, T-1, N, num_classes)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # 训练或推理时最近一次前向传播的 FSQ 索引（若量化器返回）
        self.last_fsq_indices = None

        cfg_model = config['model']
        cfg_data = config['data']
        d_model = int(cfg_model['hidden_dim'])
        T_minus_1 = int(cfg_data['sequence_length']) - 1
        N = int(cfg_data['num_points'])
        d = int(cfg_model['encoder_sequence_len'])

        # 1) 输入投影: (..., 2) -> (..., d_model)
        self.input_projection = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, T_minus_1 * N, d_model))

        # 2) Encoder (Causal)
        self.encoder = KeypointEncoder(
            num_layers=int(cfg_model['num_layers']),
            hidden_dim=d_model,
            num_heads=int(cfg_model['num_heads']),
            dropout=float(cfg_model['dropout']),
        )

        # 3) 将 Encoder 输出在序列维度上投影到长度 d
        self.encoder_output_projection = nn.Linear(T_minus_1 * N, d)

        # 4) FSQ 量化
        self.quantizer = FSQ(levels=cfg_model['fsq_levels'], dim=d_model)
        # 检测量化器是否支持 return_indices 形参（不同版本行为不同）
        support_ri = False
        for fn in (getattr(self.quantizer, 'forward', None), getattr(self.quantizer, '__call__', None)):
            if fn is None:
                continue
            try:
                if 'return_indices' in inspect.signature(fn).parameters:
                    support_ri = True
                    break
            except Exception:
                pass
        self._quantizer_support_return_indices = support_ri

        # 5) Decoder (Cross-Attention)
        self.decoder = KeypointDecoder(
            num_layers=int(cfg_model['num_layers']),
            hidden_dim=d_model,
            num_heads=int(cfg_model['num_heads']),
            dropout=float(cfg_model['dropout']),
        )

        # 6) 可学习的解码查询，对应 N 个关键点
        self.decoder_queries = nn.Parameter(torch.randn(1, N, d_model))

        # 7) 输出投影到分类 logits
        self.output_projection = nn.Linear(d_model, int(cfg_model['num_classes']))

        # Causal Mask for Encoder: (S, S), where S = (T-1)*N
        S = T_minus_1 * N
        mask = nn.Transformer.generate_square_subsequent_mask(S)
        self.register_buffer('causal_mask', mask)

    def forward(self, velocities: torch.Tensor) -> torch.Tensor:
        # velocities: (B, T-1, N, 2)
        B, T_minus_1, N, _ = velocities.shape

        # 展平时间与点维度，线性投影 + 位置编码
        x = velocities.reshape(B, T_minus_1 * N, 2)
        x = self.input_projection(x)
        x = x + self.pos_embed  # (1, S, D) broadcast to (B, S, D)

        # Encoder with causal mask (move mask to the same device as x)
        encoded = self.encoder(x, mask=self.causal_mask.to(x.device))

        # 在序列维度上投影到长度 d，然后再转回 (B, d, D)
        proj_in = encoded.transpose(1, 2)  # (B, D, S)
        proj_out = self.encoder_output_projection(proj_in)  # (B, D, d)
        to_quantize = proj_out.transpose(1, 2)  # (B, d, D)

        # FSQ 量化（尽量获取离散索引，并缓存到 last_fsq_indices 供外部读取）
        fsq_indices = None
        if self._quantizer_support_return_indices:
            quant_out = self.quantizer(to_quantize, return_indices=True)
        else:
            quant_out = self.quantizer(to_quantize)

        if isinstance(quant_out, (tuple, list)):
            quantized = None
            # 尝试从返回列表中分别找浮点量化结果与整型索引
            for item in quant_out:
                if isinstance(item, torch.Tensor):
                    if item.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                        quantized = item
                    elif item.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                        fsq_indices = item
            if quantized is None:
                # 兜底：取第一个张量作为量化输出
                quantized = quant_out[0] if isinstance(quant_out[0], torch.Tensor) else torch.as_tensor(quant_out[0], device=to_quantize.device)
        else:
            quantized = quant_out

        # 缓存索引（若可用）
        self.last_fsq_indices = fsq_indices

        # Decoder: 使用 N 个可学习 query，从量化的 memory 中解码
        queries = self.decoder_queries.expand(B, -1, -1)  # (B, N, D)
        decoded = self.decoder(tgt=queries, memory=quantized)  # (B, N, D)

        # 输出分类 logits
        logits_per_point = self.output_projection(decoded)  # (B, N, C)

        # 将每个点的分布复制到 T-1 个时间步上
        logits = logits_per_point.unsqueeze(1).expand(-1, T_minus_1, -1, -1)  # (B, T-1, N, C)
        return logits
