import torch
import torch.nn as nn

class BiaffineAttention(nn.Module):
    """
    Biaffine attention for scoring head-modifier pairs.
    Args:
        in_dim (int): Input dimension (from BiLSTM output).
        out_dim (int): Output dimension (1 for head selection, num_labels for label prediction).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.U = nn.Parameter(torch.Tensor(out_dim, in_dim + 1, in_dim + 1))
        nn.init.xavier_uniform_(self.U)
    def forward(self, head, dep):
        """
        Args:
            head (Tensor): (batch, seq_len, in_dim)
            dep (Tensor): (batch, seq_len, in_dim)
        Returns:
            Tensor: (batch, out_dim, seq_len, seq_len)
        """
        batch, seq_len, in_dim = head.size()
        ones = torch.ones(batch, seq_len, 1, device=head.device)
        head_ = torch.cat([head, ones], dim=-1)
        dep_ = torch.cat([dep, ones], dim=-1)
        # (batch, out_dim, seq_len, seq_len)
        scores = torch.einsum('bxi,oij,byj->boxy', head_, self.U, dep_)
        return scores 