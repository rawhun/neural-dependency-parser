from collections import Counter

class Vocab:
    def __init__(self, tokens, min_freq=1, specials=['<pad>', '<unk>']):
        counter = Counter(tokens)
        self.itos = specials + [t for t, c in counter.items() if c >= min_freq and t not in specials]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    def __len__(self):
        return len(self.itos)
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])
