import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.parser import DependencyParser
from tqdm import tqdm
from models.vocab import Vocab

# Utility: collate function for batching variable-length sentences
def collate_fn(batch):
    # Pad sequences to the same length in a batch
    max_len = max(len(x['words']) for x in batch)
    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))
    words = torch.tensor([pad(x['words'], 0) for x in batch], dtype=torch.long)
    pos = torch.tensor([pad(x['pos'], 0) for x in batch], dtype=torch.long)
    heads = torch.tensor([pad(x['heads'], -1) for x in batch], dtype=torch.long)
    labels = torch.tensor([pad(x['labels'], 0) for x in batch], dtype=torch.long)
    mask = (words != 0)
    return words, pos, heads, labels, mask

# Load vocabularies
def load_vocabs(proc_dir):
    with open(os.path.join(proc_dir, 'word_vocab.pkl'), 'rb') as f:
        word_vocab = pickle.load(f)
    with open(os.path.join(proc_dir, 'pos_vocab.pkl'), 'rb') as f:
        pos_vocab = pickle.load(f)
    with open(os.path.join(proc_dir, 'label_vocab.pkl'), 'rb') as f:
        label_vocab = pickle.load(f)
    return word_vocab, pos_vocab, label_vocab

# Training loop
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc_dir = os.path.join('data', 'processed')
    # Load data
    train_data = torch.load(os.path.join(proc_dir, 'train.pt'))
    dev_data = torch.load(os.path.join(proc_dir, 'dev.pt'))
    word_vocab, pos_vocab, label_vocab = load_vocabs(proc_dir)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    # Model
    model = DependencyParser(
        vocab_sizes={'word': len(word_vocab), 'pos': len(pos_vocab)},
        emb_dims={'word': 100, 'pos': 32},
        lstm_dim=256,
        num_labels=len(label_vocab)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    loss_head = nn.CrossEntropyLoss(ignore_index=-1)
    loss_label = nn.CrossEntropyLoss(ignore_index=0)
    best_las = 0.0
    # Training epochs
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for words, pos, heads, labels, mask in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
            words, pos, heads, labels, mask = words.to(device), pos.to(device), heads.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            head_scores, label_scores = model(words, pos)
            # head_scores: (batch, seq, seq), heads: (batch, seq)
            # Debug: print shapes and head indices
            print("head_scores shape:", head_scores.shape)
            print("heads shape:", heads.shape)
            print("heads max:", heads.max().item(), "heads min:", heads.min().item())
            print("heads:", heads)
            # head_scores: (batch, seq, seq), heads: (batch, seq)
            loss_h = loss_head(head_scores.permute(0,2,1), heads)
            loss_h = loss_head(head_scores.permute(0,2,1), heads)
            # label_scores: (batch, num_labels, seq, seq), labels: (batch, seq)
            # For each token, select the predicted head index for label loss
            pred_heads = heads.clamp(min=0)
            label_scores_for_heads = label_scores.permute(0,2,3,1).gather(2, pred_heads.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,label_scores.size(1))).squeeze(2)
            loss_l = loss_label(label_scores_for_heads.permute(0,2,1), labels)
            loss = loss_h + loss_l
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}')
        # Evaluate on dev set
        uas, las = evaluate(model, dev_loader, device, label_vocab)
        print(f'Epoch {epoch} Dev UAS: {uas:.2f} LAS: {las:.2f}')
        # Save best model
        if las > best_las:
            best_las = las
            torch.save(model.state_dict(), 'best_model.pt')
            print('Best model saved.')

def evaluate(model, loader, device, label_vocab):
    model.eval()
    total, correct_head, correct_label = 0, 0, 0
    with torch.no_grad():
        for words, pos, heads, labels, mask in loader:
            words, pos, heads, labels, mask = words.to(device), pos.to(device), heads.to(device), labels.to(device), mask.to(device)
            head_scores, label_scores = model(words, pos)
            pred_heads = head_scores.argmax(-1)
            pred_labels = label_scores.permute(0,2,3,1).gather(2, pred_heads.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,label_scores.size(1))).squeeze(2).argmax(-1)
            mask = mask & (heads != -1)
            total += mask.sum().item()
            correct_head += ((pred_heads == heads) & mask).sum().item()
            correct_label += ((pred_heads == heads) & (pred_labels == labels) & mask).sum().item()
    uas = 100.0 * correct_head / total
    las = 100.0 * correct_label / total
    return uas, las

if __name__ == '__main__':
    train() 