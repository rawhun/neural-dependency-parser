import os
import pickle
import torch
from torch.utils.data import DataLoader
from models.parser import DependencyParser
from train import collate_fn, load_vocabs

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc_dir = os.path.join('data', 'processed')
    test_data = torch.load(os.path.join(proc_dir, 'test.pt'))
    word_vocab, pos_vocab, label_vocab = load_vocabs(proc_dir)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model = DependencyParser(
        vocab_sizes={'word': len(word_vocab), 'pos': len(pos_vocab)},
        emb_dims={'word': 100, 'pos': 32},
        lstm_dim=256,
        num_labels=len(label_vocab)
    ).to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    uas, las = evaluate(model, test_loader, device, label_vocab)
    print(f'Test UAS: {uas:.2f}  LAS: {las:.2f}')

if __name__ == '__main__':
    main() 