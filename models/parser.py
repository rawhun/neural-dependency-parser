import torch
import torch.nn as nn
from .bilstm import BiLSTMEncoder
from .attention import BiaffineAttention

class DependencyParser(nn.Module):
    """
    Full neural dependency parser combining embeddings, BiLSTM encoder, and biaffine attention.
    """
    def __init__(self, vocab_sizes, emb_dims, lstm_dim, num_labels):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_sizes['word'], emb_dims['word'])
        self.pos_emb = nn.Embedding(vocab_sizes['pos'], emb_dims['pos'])
        self.encoder = BiLSTMEncoder(emb_dims['word'] + emb_dims['pos'], lstm_dim)
        self.head_attn = BiaffineAttention(lstm_dim*2, 1)
        self.label_attn = BiaffineAttention(lstm_dim*2, num_labels)

    def forward(self, word_idx, pos_idx):
        word_vec = self.word_emb(word_idx)
        pos_vec = self.pos_emb(pos_idx)
        x = torch.cat([word_vec, pos_vec], dim=-1)
        enc = self.encoder(x)
        head_scores = self.head_attn(enc, enc).squeeze(1)  # (batch, seq, seq)
        label_scores = self.label_attn(enc, enc)  # (batch, num_labels, seq, seq)
        return head_scores, label_scores 