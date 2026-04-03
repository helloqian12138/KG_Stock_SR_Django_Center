import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)  # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        # 第一层的输入维度应该是gate_input_start_index，而不是d_feat
        self.layers = nn.Sequential(
            # feature layer - 输入维度为gate处理后的特征数量
            nn.Linear(gate_input_start_index, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]  # N, T, D (前15个特征)
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]  # 后6个特征
        
        # feature_gate输出的维度应该与src的特征维度匹配
        gate_weights = self.feature_gate(gate_input)  # shape: [batch_size, gate_input_dim]
        
        # 只对前gate_input_start_index个特征应用gate权重
        if gate_weights.shape[1] != src.shape[2]:
            # 如果gate输出维度与src特征维度不匹配，需要调整
            gate_weights = gate_weights[:, :src.shape[2]]  # 截取前src.shape[2]个权重
        
        src = src * torch.unsqueeze(gate_weights, dim=1)
       
        output = self.layers(src).squeeze(-1)

        return output


class MASTERModel(nn.Module):
    def __init__(self, input_size=21, d_model=64, t_nhead=4, s_nhead=4, 
                T_dropout_rate=0.2, S_dropout_rate=0.2, beta=1.0,
                gate_input_start_index=15, gate_input_end_index=21, output_size=1):
        super(MASTERModel, self).__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.d_model = d_model
        self.d_feat = input_size
        self.output_size = output_size

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.model = MASTER(
            d_feat=self.d_feat, 
            d_model=self.d_model, 
            t_nhead=self.t_nhead, 
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate, 
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index, 
            beta=self.beta
        )

        # 确保所有参数移动到指定设备
        self.to(self.device)

    def forward(self, x):
        # 验证输入和模型参数的设备一致性
        model_device = next(self.parameters()).device
        if x.device != model_device:
            raise RuntimeError(f"输入张量设备 {x.device} 与模型设备 {model_device} 不一致")

        # x shape: (batch_size, seq_length, input_size)
        output = self.model(x)
        
        # 确保输出形状为 (batch_size, output_size)
        if len(output.shape) == 1:
            output = output.unsqueeze(1)
        elif len(output.shape) == 2 and output.shape[1] != self.output_size:
            output = output[:, :self.output_size]

        return output

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
            'model_type': 'MASTER',
            'input_size': self.d_feat,
            'd_model': self.d_model,
            't_nhead': self.t_nhead,
            's_nhead': self.s_nhead,
            'output_size': self.output_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'gate_input_range': f"{self.gate_input_start_index}-{self.gate_input_end_index}",
            'beta': self.beta
        }

        return info

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
        print(f"MASTER模型检查点已保存到: {filepath}")

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

        print(f"MASTER模型检查点已从 {filepath} 加载")
        print(f"Epoch: {epoch}, Loss: {loss}")

        return epoch, loss