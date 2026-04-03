import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.rnn = nn.RNN(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
        # 确保所有参数初始化时移动到指定设备
        self.to(self.device)
    
    def forward(self, x):
        # 验证输入和模型参数的设备一致性
        model_device = next(self.parameters()).device
        if x.device != model_device:
            raise RuntimeError(f"输入张量设备 {x.device} 与模型设备 {model_device} 不一致")
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, hn = self.rnn(x, h0)
        out = self.dropout(out[:, -1, :])
        out = torch.relu(self.fc(out))
        out = self.out(out)
        return out