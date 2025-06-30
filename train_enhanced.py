import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models.enhanced_parser import EnhancedDependencyParser
from train import collate_fn, load_vocabs, evaluate

def create_char_vocab(word_vocab, min_char_freq=2):
    """Create character vocabulary from word vocabulary."""
    char_counter = {}
    for word in word_vocab.itos:
        for char in word:
            char_counter[char] = char_counter.get(char, 0) + 1
    
    # Filter by frequency
    char_vocab = ['<pad>', '<unk>'] + [char for char, count in char_counter.items() if count >= min_char_freq]
    char_stoi = {char: i for i, char in enumerate(char_vocab)}
    
    return char_vocab, char_stoi

def word_to_char_ids(word, char_stoi, max_word_len=20):
    """Convert word to character indices."""
    char_ids = [char_stoi.get(char, char_stoi['<unk>']) for char in word[:max_word_len-1]]
    char_ids += [char_stoi['<pad>']] * (max_word_len - len(char_ids))
    return char_ids

def enhanced_collate_fn(batch, char_stoi, word_vocab, max_word_len=20):
    """Enhanced collate function that includes character indices."""
    # Get the original collated data
    max_len = max(len(x['words']) for x in batch)
    
    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))
    
    words = torch.tensor([pad(x['words'], 0) for x in batch], dtype=torch.long)
    pos = torch.tensor([pad(x['pos'], 0) for x in batch], dtype=torch.long)
    heads = torch.tensor([pad(x['heads'], -1) for x in batch], dtype=torch.long)
    labels = torch.tensor([pad(x['labels'], 0) for x in batch], dtype=torch.long)
    mask = (words != 0)
    
    # Create character indices
    char_ids = []
    for x in batch:
        sentence_chars = []
        for word_idx in x['words']:
            word = word_vocab.itos[word_idx] if word_idx < len(word_vocab.itos) else '<unk>'
            sentence_chars.append(word_to_char_ids(word, char_stoi, max_word_len))
        # Pad sentence
        while len(sentence_chars) < max_len:
            sentence_chars.append([char_stoi['<pad>']] * max_word_len)
        char_ids.append(sentence_chars)
    
    char_ids = torch.tensor(char_ids, dtype=torch.long)
    
    return words, pos, heads, labels, mask, char_ids

def train_enhanced_model(model_type='enhanced', use_char_emb=True, use_pretrained=False):
    """Train an enhanced dependency parser."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc_dir = os.path.join('data', 'processed')
    
    # Load data and vocabularies
    train_data = torch.load(os.path.join(proc_dir, 'train.pt'))
    dev_data = torch.load(os.path.join(proc_dir, 'dev.pt'))
    word_vocab, pos_vocab, label_vocab = load_vocabs(proc_dir)
    
    # Create character vocabulary if needed
    char_vocab, char_stoi = None, None
    if use_char_emb:
        char_vocab, char_stoi = create_char_vocab(word_vocab)
        print(f"Created character vocabulary with {len(char_vocab)} characters")
    
    # Create data loaders
    if use_char_emb:
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, 
                                collate_fn=lambda x: enhanced_collate_fn(x, char_stoi, word_vocab))
        dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False, 
                              collate_fn=lambda x: enhanced_collate_fn(x, char_stoi, word_vocab))
    else:
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
        dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    if model_type == 'enhanced':
        model = EnhancedDependencyParser(
            vocab_sizes={'word': len(word_vocab), 'pos': len(pos_vocab), 'char': len(char_vocab) if char_vocab else 0},
            emb_dims={'word': 100, 'pos': 32, 'char': 50, 'char_hidden': 100},
            lstm_dim=256,
            num_labels=len(label_vocab),
            use_char_emb=use_char_emb,
            use_pretrained=use_pretrained
        ).to(device)
    else:
        from models.parser import DependencyParser
        model = DependencyParser(
            vocab_sizes={'word': len(word_vocab), 'pos': len(pos_vocab)},
            emb_dims={'word': 100, 'pos': 32},
            lstm_dim=256,
            num_labels=len(label_vocab)
        ).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_head = nn.CrossEntropyLoss(ignore_index=-1)
    loss_label = nn.CrossEntropyLoss(ignore_index=0)
    
    best_las = 0.0
    
    # Training loop
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            if use_char_emb:
                words, pos, heads, labels, mask, char_ids = batch
                words, pos, heads, labels, mask, char_ids = words.to(device), pos.to(device), heads.to(device), labels.to(device), mask.to(device), char_ids.to(device)
            else:
                words, pos, heads, labels, mask = batch
                words, pos, heads, labels, mask = words.to(device), pos.to(device), heads.to(device), labels.to(device), mask.to(device)
                char_ids = None
            
            optimizer.zero_grad()
            
            if use_char_emb:
                head_scores, label_scores = model(words, pos, char_ids)
            else:
                head_scores, label_scores = model(words, pos)
            
            # Head loss
            loss_h = loss_head(head_scores.permute(0,2,1), heads)
            
            # Label loss
            pred_heads = heads.clamp(min=0)
            label_scores_for_heads = label_scores.permute(0,2,3,1).gather(
                2, pred_heads.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,label_scores.size(1))
            ).squeeze(2)
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
            model_name = f'best_model_{model_type}'
            if use_char_emb:
                model_name += '_char'
            if use_pretrained:
                model_name += '_pretrained'
            torch.save(model.state_dict(), f'{model_name}.pt')
            print(f'Best model saved as {model_name}.pt')
    
    return best_las

def main():
    """Main function to train enhanced models."""
    print("Training Enhanced Dependency Parser")
    print("="*50)
    
    # Train different model variants
    results = {}
    
    # 1. Baseline model
    print("\n1. Training baseline model...")
    results['baseline'] = train_enhanced_model(model_type='baseline', use_char_emb=False)
    
    # 2. Enhanced model with character embeddings
    print("\n2. Training enhanced model with character embeddings...")
    results['enhanced_char'] = train_enhanced_model(model_type='enhanced', use_char_emb=True)
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    for model_type, las in results.items():
        print(f"{model_type}: LAS = {las:.2f}")
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\n🏆 Best model: {best_model[0]} with LAS = {best_model[1]:.2f}")

if __name__ == "__main__":
    main() 