import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import pickle
from collections import Counter, defaultdict
from conllu import parse_incr
import torch
from models.vocab import Vocab

RAW_DIR = os.path.join(os.path.dirname(__file__), 'raw')
PROC_DIR = os.path.join(os.path.dirname(__file__), 'processed')


# Read .conllu file and yield sentences
def read_conllu(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for tokenlist in parse_incr(f):
            words = [t['form'] for t in tokenlist if isinstance(t['id'], int)]
            pos = [t['upostag'] for t in tokenlist if isinstance(t['id'], int)]
            heads = [t['head'] for t in tokenlist if isinstance(t['id'], int)]
            labels = [t['deprel'] for t in tokenlist if isinstance(t['id'], int)]
            yield words, pos, heads, labels

# Build vocabularies from all splits
def build_vocabs(filepaths):
    word_tokens, pos_tokens, label_tokens = [], [], []
    for fp in filepaths:
        for words, pos, _, labels in read_conllu(fp):
            word_tokens.extend(words)
            pos_tokens.extend(pos)
            label_tokens.extend(labels)
    word_vocab = Vocab(word_tokens, min_freq=2)
    pos_vocab = Vocab(pos_tokens)
    label_vocab = Vocab(label_tokens, specials=['<pad>', '<unk>'])
    return word_vocab, pos_vocab, label_vocab

# Convert sentences to indices
def encode_sentence(words, pos, heads, labels, word_vocab, pos_vocab, label_vocab):
    word_idx = [word_vocab[w] for w in words]
    pos_idx = [pos_vocab[p] for p in pos]
    seq_len = len(words)
    # Only allow head indices in [-1, 0, ..., seq_len-1]
    head_idx = []
    for h in heads:
        if h == 0:
            head_idx.append(-1)
        elif 1 <= h <= seq_len:
            head_idx.append(h-1)
        else:
            head_idx.append(-1)  # treat out-of-bounds as ignore
    label_idx = [label_vocab[l] for l in labels]
    return word_idx, pos_idx, head_idx, label_idx

# Process and save data
def process_and_save(split, filepath, word_vocab, pos_vocab, label_vocab):
    data = []
    for words, pos, heads, labels in read_conllu(filepath):
        word_idx, pos_idx, head_idx, label_idx = encode_sentence(
            words, pos, heads, labels, word_vocab, pos_vocab, label_vocab)
        data.append({
            'words': word_idx,
            'pos': pos_idx,
            'heads': head_idx,
            'labels': label_idx
        })
    out_path = os.path.join(PROC_DIR, f'{split}.pt')
    torch.save(data, out_path)
    print(f'Saved {split} set: {len(data)} sentences to {out_path}')

# Main preprocessing function
if __name__ == '__main__':
    os.makedirs(PROC_DIR, exist_ok=True)
    # Find .conllu files
    train_fp = glob.glob(os.path.join(RAW_DIR, '**', '*train.conllu'), recursive=True)[0]
    dev_fp = glob.glob(os.path.join(RAW_DIR, '**', '*dev.conllu'), recursive=True)[0]
    test_fp = glob.glob(os.path.join(RAW_DIR, '**', '*test.conllu'), recursive=True)[0]
    # Build vocabs
    word_vocab, pos_vocab, label_vocab = build_vocabs([train_fp, dev_fp, test_fp])
    # Save vocabs
    with open(os.path.join(PROC_DIR, 'word_vocab.pkl'), 'wb') as f:
        pickle.dump(word_vocab, f)
    with open(os.path.join(PROC_DIR, 'pos_vocab.pkl'), 'wb') as f:
        pickle.dump(pos_vocab, f)
    with open(os.path.join(PROC_DIR, 'label_vocab.pkl'), 'wb') as f:
        pickle.dump(label_vocab, f)
    # Process splits
    process_and_save('train', train_fp, word_vocab, pos_vocab, label_vocab)
    process_and_save('dev', dev_fp, word_vocab, pos_vocab, label_vocab)
    process_and_save('test', test_fp, word_vocab, pos_vocab, label_vocab)
    print('Preprocessing complete.') 