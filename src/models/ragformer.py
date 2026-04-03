# src/models/ragformer.py （优化后版本，性能起飞）
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RAGFormerModel(nn.Module):
    def __init__(self, input_size=21, d_model=128, nhead=8, num_layers=4, k_neighbors=7, dropout=0.3, dim_feedforward=512):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.k_neighbors = k_neighbors
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # 1. 输入投影：保持原始特征 + 检索特征分开处理
        self.input_proj = nn.Linear(input_size, d_model)
        self.retrieved_proj = nn.Linear(input_size, d_model)

        # 2. 位置编码（修复！必须对整个序列加）
        self.pos_encoder = PositionalEncoding(d_model, max_len=60)

        # 3. Transformer 编码器（加强！）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 输出头
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 拼接原始 + 增强
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.historical_data = None
        self.to(self.device)

    def set_historical_data(self, historical_data):
        if historical_data.shape[1] != self.input_size:
            raise ValueError(f"historical_data 维度应为 {self.input_size}，实际为 {historical_data.shape[1]}")
        self.historical_data = historical_data.to(self.device)

    def retrieve(self, x):
        """返回 top-k 相似样本的加权平均嵌入（不拼接原始特征）"""
        if self.historical_data is None:
            raise ValueError("historical_data 未设置")

        # x: (B, T, D), 取最后一天
        query = x[:, -1, :]  # (B, D)
        query = query.unsqueeze(1)  # (B, 1, D)

        # 余弦相似度
        sims = F.cosine_similarity(query, self.historical_data.unsqueeze(0), dim=-1)  # (B, N)
        topk_sim, topk_idx = torch.topk(sims, k=self.k_neighbors, dim=-1)  # (B, K)
        weights = F.softmax(topk_sim / 0.1, dim=-1).unsqueeze(-1)  # (B, K, 1)

        # 加权平均
        retrieved = self.historical_data[topk_idx]  # (B, K, D)
        retrieved = (retrieved * weights).sum(dim=1)  # (B, D)
        return retrieved  # (B, D)

    def forward(self, x):
        # x: (B, T, 21)
        B, T, D = x.shape

        # 原始序列编码
        src = self.input_proj(x)  # (B, T, d_model)
        src = self.pos_encoder(src)
        memory = self.transformer(src)  # (B, T, d_model)
        last_memory = memory[:, -1, :]  # (B, d_model)

        # 检索增强
        retrieved = self.retrieve(x)  # (B, D)
        retrieved = self.retrieved_proj(retrieved)  # (B, d_model)

        # 融合：拼接 + 全连接（最稳）
        fused = torch.cat([last_memory, retrieved], dim=-1)  # (B, d_model*2)
        out = self.fc_out(fused)  # (B, 1)

        return out.squeeze(-1).unsqueeze(-1)  # (B, 1)