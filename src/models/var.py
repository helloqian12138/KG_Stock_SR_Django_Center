import torch
import torch.nn as nn
import numpy as np
from src.models.lstm import LSTMModel
from scipy.stats import norm

class VaRModel(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=2, confidence_level=0.95, dropout=0.2):
        super(VaRModel, self).__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.lstm = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.confidence_level = confidence_level
        # 确保所有参数移动到指定设备
        self.to(self.device)
    
    def calculate_var(self, predictions):
        # 展平为 (batch_size, -1) 并沿第 1 维排序
        batch_size = predictions.size(0)
        flattened_predictions = predictions.view(batch_size, -1)  # 展平
        sorted_predictions = torch.sort(flattened_predictions, dim=1)[0]  # 沿第 1 维排序
        var_index = int((1 - self.confidence_level) * flattened_predictions.size(1))
        var_values = sorted_predictions[:, var_index:var_index + 1]  # 直接取切片，保留梯度
        return var_values  # 返回 (batch_size, 1) 张量
    
    def forward(self, x):
        # 验证输入和模型参数的设备一致性
        model_device = next(self.parameters()).device
        if x.device != model_device:
            raise RuntimeError(f"输入张量设备 {x.device} 与模型设备 {model_device} 不一致")
        
        # x 应为 (batch_size, sequence_length, input_size)
        lstm_output = self.lstm(x)  # 假设 LSTM 输出 (batch_size, output_size)
        print(f"LSTM output requires_grad: {lstm_output.requires_grad}, device: {lstm_output.device}")
        var_values = self.calculate_var(lstm_output)
        print(f"VaR output requires_grad: {var_values.requires_grad}, device: {var_values.device}")
        return var_values  # 返回 (batch_size, 1)
    
    def predict(self, x):
        # 仅在评估模式下使用 no_grad
        if self.training:
            return self.forward(x)  # 训练时保留梯度
        else:
            with torch.no_grad():
                return self.forward(x)  # 评估时禁用梯度