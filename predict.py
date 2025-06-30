import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pickle
import spacy
from models.parser import DependencyParser
from models.vocab import Vocab
from train import load_vocabs

def load_model_and_vocabs():
    """Load the trained model and vocabularies."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc_dir = os.path.join('data', 'processed')
    
    # Load vocabularies
    word_vocab, pos_vocab, label_vocab = load_vocabs(proc_dir)
    
    # Initialize model
    model = DependencyParser(
        vocab_sizes={'word': len(word_vocab), 'pos': len(pos_vocab)},
        emb_dims={'word': 100, 'pos': 32},
        lstm_dim=256,
        num_labels=len(label_vocab)
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()
    
    return model, word_vocab, pos_vocab, label_vocab, device

def tokenize_sentence(sentence):
    """Tokenize and POS tag a sentence using spaCy."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy English model not found. Installing...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(sentence)
    words = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    return words, pos_tags

def predict_dependencies(model, words, pos_tags, word_vocab, pos_vocab, label_vocab, device):
    """Predict dependency heads and labels for a sentence."""
    # Convert to indices
    word_idx = [word_vocab.get(w, word_vocab['<unk>']) for w in words]
    pos_idx = [pos_vocab.get(p, pos_vocab['<unk>']) for p in pos_tags]
    
    # Convert to tensors
    word_tensor = torch.tensor([word_idx], dtype=torch.long).to(device)
    pos_tensor = torch.tensor([pos_idx], dtype=torch.long).to(device)
    
    # Predict
    with torch.no_grad():
        head_scores, label_scores = model(word_tensor, pos_tensor)
        
        # Get predictions
        pred_heads = head_scores.argmax(-1).squeeze(0)  # (seq_len,)
        pred_labels = label_scores.permute(0,2,3,1).gather(
            2, pred_heads.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,label_scores.size(1))
        ).squeeze(2).argmax(-1).squeeze(0)  # (seq_len,)
    
    return pred_heads.cpu().numpy(), pred_labels.cpu().numpy()

def format_dependency_tree(words, pos_tags, heads, labels, label_vocab):
    """Format the dependency tree in a readable way."""
    print("\nDependency Tree:")
    print("-" * 50)
    print(f"{'ID':<3} {'Word':<15} {'POS':<8} {'Head':<8} {'Label':<15}")
    print("-" * 50)
    
    for i, (word, pos, head, label) in enumerate(zip(words, pos_tags, heads, labels)):
        head_word = words[head] if head < len(words) else "ROOT"
        label_name = label_vocab.itos[label] if label < len(label_vocab.itos) else "UNK"
        print(f"{i:<3} {word:<15} {pos:<8} {head_word:<8} {label_name:<15}")

def main():
    """Main function to parse sentences."""
    print("Loading model and vocabularies...")
    model, word_vocab, pos_vocab, label_vocab, device = load_model_and_vocabs()
    print("Model loaded successfully!")
    
    # Example sentences
    test_sentences = [
        "The cat sat on the mat.",
        "I love neural networks.",
        "She quickly ran to the store.",
        "The beautiful red car drove fast."
    ]
    
    print(f"\nParsing {len(test_sentences)} example sentences...")
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{'='*60}")
        print(f"Sentence {i}: {sentence}")
        print(f"{'='*60}")
        
        # Tokenize
        words, pos_tags = tokenize_sentence(sentence)
        print(f"Tokens: {words}")
        print(f"POS tags: {pos_tags}")
        
        # Predict dependencies
        heads, labels = predict_dependencies(model, words, pos_tags, word_vocab, pos_vocab, label_vocab, device)
        
        # Format and display
        format_dependency_tree(words, pos_tags, heads, labels, label_vocab)
    
    # Interactive mode
    print(f"\n{'='*60}")
    print("Interactive Mode - Enter your own sentences (type 'quit' to exit):")
    print(f"{'='*60}")
    
    while True:
        user_input = input("\nEnter a sentence: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
            
        try:
            words, pos_tags = tokenize_sentence(user_input)
            heads, labels = predict_dependencies(model, words, pos_tags, word_vocab, pos_vocab, label_vocab, device)
            format_dependency_tree(words, pos_tags, heads, labels, label_vocab)
        except Exception as e:
            print(f"Error parsing sentence: {e}")

if __name__ == "__main__":
    main() 