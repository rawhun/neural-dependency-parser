import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    """
    Bi-directional LSTM encoder for contextual token representations.
    Args:
        input_dim (int): Dimension of input embeddings.
        hidden_dim (int): Hidden size of the LSTM.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate between LSTM layers.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.33):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): (batch, seq_len, input_dim)
            mask (Tensor, optional): (batch, seq_len) mask for padding (not used here)
        Returns:
            Tensor: (batch, seq_len, 2*hidden_dim)
        """
        out, _ = self.lstm(x)
        return out 