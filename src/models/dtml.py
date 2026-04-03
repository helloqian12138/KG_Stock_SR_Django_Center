import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class AttLstm(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu'):
        super(AttLstm, self).__init__()
        self.lstm_hidden_layer = hidden_size
        self.device = device
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, _input):
        # initiate hidden states
        batch_size = _input.size(0)
        device = _input.device  # 使用输入张量的设备
        h_state = torch.zeros((batch_size, self.lstm_hidden_layer), device=device)
        c_state = torch.zeros((batch_size, self.lstm_hidden_layer), device=device)
        h_states = torch.zeros((0, batch_size, self.lstm_hidden_layer), device=device)

        # iterate time series
        for j in range(_input.size(1)):
            h_state, c_state = self.lstm_cell(_input[:, j, :], (h_state, c_state))
            h_states = torch.cat((h_states, h_state.unsqueeze(0)))
        h_states = h_states.transpose(0, 1)

        # calculate attention value(context_vector)
        att_score = torch.matmul(h_states, h_state.view(-1, self.lstm_hidden_layer, 1))
        att_dist = att_score / torch.sum(att_score, dim=1).view(-1, 1, 1)
        context_vector = torch.matmul(att_dist.view(batch_size, 1, -1), h_states)

        return context_vector


class DTMLModel(nn.Module):
    """
    DTML (Deep Time-series Multi-Level) Model for financial prediction
    
    Args:
        input_size: Number of input features
        n_stock: Number of stocks (default: 1 for single time series)
        n_time: Sequence length (window size)
        n_heads: Number of attention heads
        d_lstm_input: LSTM input dimension (if None, uses input_size)
        lstm_hidden_layer: LSTM hidden layer size
        output_size: Output dimension (default: 1 for regression)
    """
    def __init__(self, input_size, n_stock=1, n_time=60, n_heads=4, d_lstm_input=None, 
                 lstm_hidden_layer=64, output_size=1, device='cpu'):
        super(DTMLModel, self).__init__()
        if d_lstm_input is None:
            d_lstm_input = input_size
        
        self.input_size = input_size
        self.n_stock = n_stock
        self.n_time = n_time
        self.lstm_hidden_layer = lstm_hidden_layer
        self.device = device
        self.output_size = output_size
        self.att_weight = None

        # Feature Transformation layers
        self.stock_f_tr_layer = nn.Sequential(
            nn.Linear(input_size, d_lstm_input),
            nn.Tanh()
        )
        self.macro_f_tr_layer = nn.Sequential(
            nn.Linear(input_size, d_lstm_input),  # Using same input_size for macro
            nn.Tanh()
        )

        # LSTM cell
        self.stock_att_lstm = AttLstm(input_size=d_lstm_input, hidden_size=lstm_hidden_layer, device=device)
        self.macro_att_lstm = AttLstm(input_size=d_lstm_input, hidden_size=lstm_hidden_layer, device=device)

        # Context Normalization parameters
        self.norm_weight = nn.Parameter(torch.randn(n_stock, lstm_hidden_layer))
        self.norm_bias = nn.Parameter(torch.randn(n_stock, lstm_hidden_layer))

        # Macro weight for Multi-Level Contexts
        self.macro_weight = nn.Parameter(torch.randn(1))

        # Multi-head attention
        self.multi_head_att = nn.MultiheadAttention(lstm_hidden_layer, n_heads, batch_first=True)

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_layer, lstm_hidden_layer * 4),
            nn.ReLU(),
            nn.Linear(lstm_hidden_layer * 4, lstm_hidden_layer)
        )

        # Final layer for regression (single output)
        self.final_layer = nn.Linear(lstm_hidden_layer, output_size)

    def forward(self, x):
        """
        Forward pass for DTML model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # 添加调试信息和错误处理
        print(f"DTML input shape: {x.shape}, ndim: {x.ndim}")
        
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor (batch_size, seq_len, input_size), got {x.ndim}D tensor with shape {x.shape}")
        
        batch_size, seq_len, input_dim = x.shape
        
        # For simplicity, we'll treat the input as both stock and macro data
        # In a real implementation, you might want to split the input
        stock_input = x.unsqueeze(1)  # Add stock dimension: (batch, 1, seq, features)
        macro_input = x  # Keep as (batch, seq, features)
        
        # Feature Transformation
        stock_input = self.stock_f_tr_layer(stock_input)
        macro_input = self.macro_f_tr_layer(macro_input)

        # Attention LSTM
        c_matrix = torch.zeros((batch_size, 0, self.lstm_hidden_layer), device=x.device)

        # Process stock data
        for i in range(stock_input.size(1)):
            context_vector = self.stock_att_lstm(stock_input[:, i, :, :])
            c_matrix = torch.cat((c_matrix, context_vector), dim=1)
        
        # Process macro data
        macro_context = self.macro_att_lstm(macro_input)

        # Context Normalization
        mean_val = torch.mean(c_matrix, dim=(1, 2), keepdim=True)
        std_val = torch.std(c_matrix, dim=(1, 2), keepdim=True) + 1e-8
        c_matrix = self.norm_weight.unsqueeze(0) * ((c_matrix - mean_val) / std_val) + self.norm_bias.unsqueeze(0)

        # Multi-Level Contexts
        ml_c_matrix = c_matrix + self.macro_weight * macro_context

        # Multi-Head Self-Attention
        att_value_matrix, self.att_weight = self.multi_head_att(ml_c_matrix, ml_c_matrix, ml_c_matrix)

        # Nonlinear Transformation
        mlp_out_matrix = torch.zeros_like(att_value_matrix, device=x.device)
        for i in range(att_value_matrix.size(1)):
            mlp_out = self.mlp(ml_c_matrix[:, i, :] + att_value_matrix[:, i, :])
            mlp_out_matrix[:, i, :] = mlp_out
        
        out_matrix = torch.tanh(ml_c_matrix + att_value_matrix + mlp_out_matrix)

        # Final Prediction - average pooling across stocks
        pooled_output = torch.mean(out_matrix, dim=1)  # (batch_size, lstm_hidden_layer)
        final_output = self.final_layer(pooled_output)  # (batch_size, output_size)

        return final_output.squeeze(-1)  # Remove last dimension if output_size=1

    def predict(self, x):
        """
        Prediction method for compatibility with the prediction pipeline
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, sequence_length) for multi-step prediction
        """
        print(f"DTML predict called with input shape: {x.shape}")
        
        # DTML 默认输出单步预测，需要扩展为多步
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor (batch_size, seq_len, input_size), got {x.ndim}D tensor with shape {x.shape}")
        
        batch_size, seq_len, input_dim = x.shape
        
        # 对每个时间步进行预测
        predictions = []
        for t in range(seq_len):
            # 使用当前时间步及之前的数据进行预测
            current_input = x[:, :t+1, :]  # (batch_size, t+1, input_dim)
            
            # 如果序列太短，至少使用最后一个时间步
            if current_input.size(1) == 0:
                current_input = x[:, :1, :]
            
            # 调用 forward 方法进行单步预测
            pred = self.forward(current_input)  # (batch_size,)
            predictions.append(pred.unsqueeze(1))  # (batch_size, 1)
        
        # 拼接所有预测结果
        output = torch.cat(predictions, dim=1)  # (batch_size, seq_len)
        print(f"DTML predict output shape: {output.shape}")
        return output

    def save_model(self, filepath):
        """Save model state dict to file"""
        torch.save(self.state_dict(), filepath)
        print(f"DTML model saved to {filepath}")

    def load_model(self, filepath):
        """Load model state dict from file"""
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"DTML model loaded from {filepath}")

    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': 'DTML',
            'input_size': self.input_size,
            'n_stock': self.n_stock,
            'n_time': self.n_time,
            'lstm_hidden_layer': self.lstm_hidden_layer,
            'output_size': self.output_size,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }