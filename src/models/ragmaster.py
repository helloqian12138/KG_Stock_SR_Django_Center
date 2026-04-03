import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """空间注意力机制 - 来自MASTER"""
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
    """时间注意力机制 - 来自MASTER"""
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
    """特征门控机制 - 来自MASTER"""
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        if gate_input.numel() == 0 or gate_input.size(0) == 0:
            device = self.trans.weight.device
            return torch.full((gate_input.size(0), self.d_output), 
                            1.0 / self.d_output, device=device)
        
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):
    """时间注意力聚合 - 来自MASTER"""
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


class RetrievalAugmentedLayer(nn.Module):
    """检索增强层 - 来自RAGFormer的核心思想"""
    def __init__(self, input_size, k_neighbors=5):
        super().__init__()
        self.input_size = input_size
        self.k_neighbors = k_neighbors
        self.historical_data = None
        
        # 用于融合检索到的信息的网络
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_size * (k_neighbors + 1), input_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size * 2, input_size),
            nn.LayerNorm(input_size)
        )

    def set_historical_data(self, historical_data):
        if historical_data.shape[1] != self.input_size:
            raise ValueError(f"historical_data 特征维度应为 {self.input_size}，实际为 {historical_data.shape[1]}")
        self.historical_data = historical_data.to(next(self.parameters()).device)   

    def retrieve_similar(self, x):
        """检索相似的历史数据"""
        if self.historical_data is None:
            # 如果没有历史数据，返回原始输入
            return x[:, -1, :]
        
        # x: (batch_size, seq_len, input_size)
        x_last = x[:, -1, :]  # (batch_size, input_size)
        batch_size = x_last.size(0)
        
        # 计算余弦相似性
        x_last_norm = x_last.unsqueeze(1)  # (batch_size, 1, input_size)
        historical_norm = self.historical_data.unsqueeze(0)  # (1, num_samples, input_size)
        similarities = F.cosine_similarity(x_last_norm, historical_norm, dim=-1)  # (batch_size, num_samples)
        
        # 获取 top-k 索引
        top_k_values, top_k_indices = torch.topk(similarities, k=self.k_neighbors, dim=-1, largest=True)
        weights = F.softmax(top_k_values, dim=-1)  # (batch_size, k_neighbors)
        
        # 获取相似序列
        similar_sequences = self.historical_data[top_k_indices]  # (batch_size, k_neighbors, input_size)
        
        # 计算加权相似序列
        weighted_sequences = (similar_sequences * weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, input_size)
        
        # 拼接原始特征和检索到的特征
        x_augmented = torch.cat([x_last] + [weighted_sequences] * self.k_neighbors, dim=-1)  # (batch_size, input_size * (k_neighbors + 1))
        
        return x_augmented
    
    def forward(self, x):
        """前向传播"""
        # 检索增强
        x_augmented = self.retrieve_similar(x)
        
        # 融合检索信息
        x_fused = self.fusion_layer(x_augmented)
        
        return x_fused


class RAGMaster(nn.Module):
    """RAGMaster核心模型 - 结合MASTER和RAGFormer的优势"""
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, 
                 gate_input_start_index, gate_input_end_index, beta, k_neighbors=5):
        super(RAGMaster, self).__init__()
        
        # 检索增强层
        self.retrieval_layer = RetrievalAugmentedLayer(gate_input_start_index, k_neighbors)
        
        # 特征门控机制
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, gate_input_start_index, beta=beta)
        
        # 主要的注意力层序列
        self.layers = nn.Sequential(
            # 特征投影层
            nn.Linear(gate_input_start_index, d_model),
            PositionalEncoding(d_model),
            # 时间内注意力聚合
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # 股票间注意力聚合  
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # 时间注意力聚合
            TemporalAttention(d_model=d_model),
            # 解码器
            nn.Linear(d_model, 1)
        )

    def set_historical_data(self, historical_data):
        """设置历史数据用于检索增强"""
        self.retrieval_layer.set_historical_data(historical_data)

    def forward(self, x):
        # 分离基础特征和门控特征
        src = x[:, :, :self.gate_input_start_index]  # N, T, D (前15个特征)
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]  # 后6个特征
        
        # 检索增强处理
        src_enhanced = self.retrieval_layer(src)  # 应用检索增强到基础特征
        
        # 特征门控
        gate_weights = self.feature_gate(gate_input)  # shape: [batch_size, gate_input_start_index]
        
        # 确保gate权重维度匹配
        if gate_weights.shape[1] != src_enhanced.shape[1]:
            gate_weights = gate_weights[:, :src_enhanced.shape[1]]
        
        # 应用门控权重到增强特征
        src_gated = src_enhanced * gate_weights
        
        # 通过主要的注意力层
        output = self.layers(src_gated.unsqueeze(1)).squeeze(-1)  # 添加时间维度然后移除

        return output


class RAGMasterModel(nn.Module):
    """RAGMaster完整模型包装器"""
    def __init__(self, input_size=21, d_model=64, t_nhead=4, s_nhead=4, 
                 T_dropout_rate=0.2, S_dropout_rate=0.2, beta=1.0,
                 gate_input_start_index=15, gate_input_end_index=21, 
                 k_neighbors=5, output_size=1):
        super(RAGMasterModel, self).__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.d_model = d_model
        self.d_feat = input_size
        self.output_size = output_size
        self.k_neighbors = k_neighbors

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.model = RAGMaster(
            d_feat=self.d_feat, 
            d_model=self.d_model, 
            t_nhead=self.t_nhead, 
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate, 
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index, 
            beta=self.beta,
            k_neighbors=self.k_neighbors
        )

        # 确保所有参数移动到指定设备
        self.to(self.device)

    def set_historical_data(self, historical_data):
        """设置历史数据用于检索增强"""
        if historical_data.shape[1] != self.gate_input_start_index:
            raise ValueError(f"historical_data 特征维度应为 {self.gate_input_start_index}，实际为 {historical_data.shape[1]}")
        self.model.set_historical_data(historical_data.to(self.device))

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
        """预测方法，根据训练状态决定是否使用梯度"""
        if self.training:
            return self.forward(x)  # 训练时保留梯度
        else:
            with torch.no_grad():
                return self.forward(x)  # 评估时禁用梯度

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'model_type': 'RAGMaster',
            'input_size': self.d_feat,
            'd_model': self.d_model,
            't_nhead': self.t_nhead,
            's_nhead': self.s_nhead,
            'k_neighbors': self.k_neighbors,
            'output_size': self.output_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'gate_input_range': f"{self.gate_input_start_index}-{self.gate_input_end_index}",
            'beta': self.beta
        }

        return info

    def save_checkpoint(self, filepath, epoch, loss, optimizer_state=None):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'loss': loss,
            'model_config': {
                'input_size': self.d_feat,
                'd_model': self.d_model,
                't_nhead': self.t_nhead,
                's_nhead': self.s_nhead,
                'T_dropout_rate': self.T_dropout_rate,
                'S_dropout_rate': self.S_dropout_rate,
                'beta': self.beta,
                'gate_input_start_index': self.gate_input_start_index,
                'gate_input_end_index': self.gate_input_end_index,
                'k_neighbors': self.k_neighbors,
                'output_size': self.output_size
            }
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, filepath)
        print(f"模型检查点已保存到: {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location='cpu')
        config = checkpoint['model_config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint['epoch'], checkpoint['loss']