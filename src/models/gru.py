import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        
        # 确保所有参数移动到指定设备
        self.to(self.device)
    
    def forward(self, x):
        # 验证输入和模型参数的设备一致性
        model_device = next(self.parameters()).device
        if x.device != model_device:
            raise RuntimeError(f"输入张量设备 {x.device} 与模型设备 {model_device} 不一致")
        
        if not x.requires_grad:
            # print(f"Input x requires_grad: {x.requires_grad}")
            x = x.clone().detach().requires_grad_(True)  # 确保输入具有梯度
        
        # x shape: (batch_size, seq_length, input_size)
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # GRU前向传播
        out, _ = self.gru(x, h0)
        # out shape: (batch_size, seq_length, hidden_size)
        
        # 我们只取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 通过全连接层
        out = self.fc(out)
        
        # print(f"GRU output requires_grad: {out.requires_grad}, device: {out.device}")
        return out
    
    def predict(self, x):
        """
        预测方法，根据训练状态决定是否使用梯度
        """
        if self.training:
            return self.forward(x)  # 训练时保留梯度
        else:
            with torch.no_grad():
                return self.forward(x)  # 评估时禁用梯度
    
    def get_model_info(self):
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'GRU',
            'input_size': self.gru.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.fc.out_features,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }
        
        return info
    
    def reset_parameters(self):
        """
        重新初始化模型参数
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        print("GRU模型参数已重新初始化")
    
    def save_checkpoint(self, filepath, epoch, loss, optimizer_state=None):
        """
        保存模型检查点
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'loss': loss,
            'model_info': self.get_model_info()
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
        print(f"GRU模型检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath, optimizer=None):
        """
        加载模型检查点
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        
        print(f"GRU模型检查点已从 {filepath} 加载")
        print(f"Epoch: {epoch}, Loss: {loss}")
        
        return epoch, loss
    
    def freeze_layers(self, freeze_gru=False, freeze_fc=False):
        """
        冻结指定层的参数
        """
        if freeze_gru:
            for param in self.gru.parameters():
                param.requires_grad = False
            print("GRU层已冻结")
        
        if freeze_fc:
            for param in self.fc.parameters():
                param.requires_grad = False
            print("全连接层已冻结")
    
    def unfreeze_layers(self):
        """
        解冻所有层的参数
        """
        for param in self.parameters():
            param.requires_grad = True
        print("所有层已解冻")
    
    def get_layer_gradients(self):
        """
        获取各层的梯度信息
        """
        gradients = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                gradients[name] = {
                    'grad_norm': grad_norm.item(),
                    'param_shape': list(param.shape),
                    'requires_grad': param.requires_grad
                }
            else:
                gradients[name] = {
                    'grad_norm': 0.0,
                    'param_shape': list(param.shape),
                    'requires_grad': param.requires_grad
                }
        
        return gradients
    
    def apply_weight_decay(self, weight_decay=1e-4):
        """
        应用权重衰减正则化
        """
        for param in self.parameters():
            if param.requires_grad and len(param.shape) >= 2:  # 只对权重矩阵应用
                param.data.mul_(1 - weight_decay)