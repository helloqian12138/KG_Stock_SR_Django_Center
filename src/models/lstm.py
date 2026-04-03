import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        # 确保所有参数移动到指定设备
        self.to(self.device)
    
    def forward(self, x):
        # 验证输入和模型参数的设备一致性
        model_device = next(self.parameters()).device
        if x.device != model_device:
            raise RuntimeError(f"输入张量设备 {x.device} 与模型设备 {model_device} 不一致")
        
        if not x.requires_grad:
            print(f"Input x requires_grad: {x.requires_grad}")
            x = x.clone().detach().requires_grad_(True)  # 确保输入具有梯度
        
        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch_size, seq_length, hidden_size)
        # 我们只取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)  # 应用sigmoid激活函数，确保输出在[0,1]范围内
        print(f"LSTM output requires_grad: {out.requires_grad}, device: {out.device}")
        return out