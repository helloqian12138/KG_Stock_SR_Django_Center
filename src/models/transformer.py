import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_size=21, d_model=64, nhead=4, num_layers=2, dropout=0.2, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.fc_in = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, device=self.device)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 启用 batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, 1)
        # 确保所有参数移动到指定设备
        self.to(self.device)
    
    def forward(self, x):
        # 验证输入和模型参数的设备一致性
        model_device = next(self.parameters()).device
        if x.device != model_device:
            raise RuntimeError(f"输入张量设备 {x.device} 与模型设备 {model_device} 不一致")
        
        x = self.fc_in(x)  # [batch, seq_len, input_size] -> [batch, seq_len, d_model]
        x = self.pos_encoder(x)  # 应用位置编码
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        x = self.fc_out(x[:, -1, :])  # 取最后一个时间步，[batch, d_model] -> [batch, 1]
        print(f"Transformer output device: {x.device}")
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe.to(self.device))  # 确保 pe 张量在指定设备
        # 确保 dropout 层移动到指定设备
        self.to(self.device)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # 动态裁剪到输入序列长度，无需再次 .to(x.device)
        return self.dropout(x)