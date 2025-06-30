import torch
import torch.nn as nn
import torch.nn.functional as F
from .bilstm import BiLSTMEncoder
from .attention import BiaffineAttention

class CharCNN(nn.Module):
    """Character-level CNN for word representations."""
    def __init__(self, char_vocab_size, char_emb_dim=50, char_hidden_dim=100, kernel_sizes=[3, 4, 5]):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_emb_dim, char_hidden_dim // len(kernel_sizes), k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, char_ids):
        # char_ids: (batch, seq_len, max_word_len)
        batch_size, seq_len, max_word_len = char_ids.size()
        
        # Reshape for CNN
        char_ids = char_ids.view(-1, max_word_len)  # (batch*seq_len, max_word_len)
        char_emb = self.char_emb(char_ids)  # (batch*seq_len, max_word_len, char_emb_dim)
        char_emb = char_emb.transpose(1, 2)  # (batch*seq_len, char_emb_dim, max_word_len)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(char_emb))  # (batch*seq_len, hidden_dim, max_word_len-k+1)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))  # (batch*seq_len, hidden_dim, 1)
            conv_out = conv_out.squeeze(2)  # (batch*seq_len, hidden_dim)
            conv_outputs.append(conv_out)
        
        # Concatenate and reshape back
        char_repr = torch.cat(conv_outputs, dim=1)  # (batch*seq_len, total_hidden_dim)
        char_repr = self.dropout(char_repr)
        char_repr = char_repr.view(batch_size, seq_len, -1)  # (batch, seq_len, total_hidden_dim)
        
        return char_repr

class EnhancedDependencyParser(nn.Module):
    """
    Enhanced dependency parser with character embeddings and pre-trained word vectors.
    """
    def __init__(self, vocab_sizes, emb_dims, lstm_dim, num_labels, 
                 use_char_emb=True, use_pretrained=False, pretrained_emb=None):
        super().__init__()
        
        self.use_char_emb = use_char_emb
        self.use_pretrained = use_pretrained
        
        # Word embeddings
        if use_pretrained and pretrained_emb is not None:
            self.word_emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=False)
        else:
            self.word_emb = nn.Embedding(vocab_sizes['word'], emb_dims['word'])
        
        # POS embeddings
        self.pos_emb = nn.Embedding(vocab_sizes['pos'], emb_dims['pos'])
        
        # Character embeddings
        if use_char_emb:
            self.char_cnn = CharCNN(
                char_vocab_size=vocab_sizes.get('char', 100),
                char_emb_dim=emb_dims.get('char', 50),
                char_hidden_dim=emb_dims.get('char_hidden', 100)
            )
            char_dim = emb_dims.get('char_hidden', 100)
        else:
            char_dim = 0
        
        # Calculate input dimension for LSTM
        input_dim = emb_dims['word'] + emb_dims['pos'] + char_dim
        
        # BiLSTM encoder
        self.encoder = BiLSTMEncoder(input_dim, lstm_dim)
        
        # Biaffine attention
        self.head_attn = BiaffineAttention(lstm_dim*2, 1)
        self.label_attn = BiaffineAttention(lstm_dim*2, num_labels)
        
        # Additional layers for better performance
        self.dropout = nn.Dropout(0.33)
        self.layer_norm = nn.LayerNorm(lstm_dim*2)
    
    def forward(self, word_idx, pos_idx, char_idx=None):
        """
        Args:
            word_idx: (batch, seq_len) word indices
            pos_idx: (batch, seq_len) POS tag indices
            char_idx: (batch, seq_len, max_word_len) character indices (optional)
        """
        # Word and POS embeddings
        word_vec = self.word_emb(word_idx)
        pos_vec = self.pos_emb(pos_idx)
        
        # Concatenate word and POS embeddings
        x = torch.cat([word_vec, pos_vec], dim=-1)
        
        # Add character embeddings if available
        if self.use_char_emb and char_idx is not None:
            char_vec = self.char_cnn(char_idx)
            x = torch.cat([x, char_vec], dim=-1)
        
        # Apply dropout and layer normalization
        x = self.dropout(x)
        
        # BiLSTM encoding
        enc = self.encoder(x)
        enc = self.layer_norm(enc)
        
        # Biaffine attention
        head_scores = self.head_attn(enc, enc).squeeze(1)  # (batch, seq, seq)
        label_scores = self.label_attn(enc, enc)  # (batch, num_labels, seq, seq)
        
        return head_scores, label_scores

class MultiTaskDependencyParser(nn.Module):
    """
    Multi-task dependency parser that also predicts POS tags.
    """
    def __init__(self, vocab_sizes, emb_dims, lstm_dim, num_labels, num_pos_tags):
        super().__init__()
        
        # Shared components
        self.word_emb = nn.Embedding(vocab_sizes['word'], emb_dims['word'])
        self.encoder = BiLSTMEncoder(emb_dims['word'], lstm_dim)
        
        # Task-specific components
        self.head_attn = BiaffineAttention(lstm_dim*2, 1)
        self.label_attn = BiaffineAttention(lstm_dim*2, num_labels)
        self.pos_classifier = nn.Linear(lstm_dim*2, num_pos_tags)
        
        self.dropout = nn.Dropout(0.33)
    
    def forward(self, word_idx):
        # Word embeddings
        word_vec = self.word_emb(word_idx)
        word_vec = self.dropout(word_vec)
        
        # Shared encoding
        enc = self.encoder(word_vec)
        
        # Task-specific predictions
        head_scores = self.head_attn(enc, enc).squeeze(1)
        label_scores = self.label_attn(enc, enc)
        pos_scores = self.pos_classifier(enc)
        
        return head_scores, label_scores, pos_scores 