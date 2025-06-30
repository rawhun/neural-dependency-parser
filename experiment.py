import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from models.parser import DependencyParser
from train import collate_fn, load_vocabs, evaluate

class ExperimentRunner:
    """Class to run hyperparameter experiments."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.proc_dir = os.path.join('data', 'processed')
        self.results = []
        
        # Load data
        print("Loading data...")
        train_data = torch.load(os.path.join(self.proc_dir, 'train.pt'))
        dev_data = torch.load(os.path.join(self.proc_dir, 'dev.pt'))
        self.word_vocab, self.pos_vocab, self.label_vocab = load_vocabs(self.proc_dir)
        
        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
        self.dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
    def train_model(self, config, max_epochs=10):
        """Train a model with given configuration."""
        print(f"\nTraining with config: {config}")
        
        # Initialize model
        model = DependencyParser(
            vocab_sizes={'word': len(self.word_vocab), 'pos': len(self.pos_vocab)},
            emb_dims={'word': config['word_emb_dim'], 'pos': config['pos_emb_dim']},
            lstm_dim=config['lstm_dim'],
            num_labels=len(self.label_vocab)
        ).to(self.device)
        
        # Optimizer
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
        
        # Loss functions
        loss_head = nn.CrossEntropyLoss(ignore_index=-1)
        loss_label = nn.CrossEntropyLoss(ignore_index=0)
        
        best_las = 0.0
        start_time = time.time()
        
        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0.0
            
            for words, pos, heads, labels, mask in self.train_loader:
                words, pos, heads, labels, mask = words.to(self.device), pos.to(self.device), heads.to(self.device), labels.to(self.device), mask.to(self.device)
                
                optimizer.zero_grad()
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
            
            # Evaluate on dev set
            uas, las = evaluate(model, self.dev_loader, self.device, self.label_vocab)
            
            print(f"Epoch {epoch}: Loss={total_loss/len(self.train_loader):.4f}, UAS={uas:.2f}, LAS={las:.2f}")
            
            if las > best_las:
                best_las = las
                # Save best model for this config
                torch.save(model.state_dict(), f'best_model_{config["name"]}.pt')
        
        training_time = time.time() - start_time
        
        return {
            'config': config,
            'best_las': best_las,
            'training_time': training_time,
            'epochs': max_epochs
        }
    
    def run_experiments(self):
        """Run multiple hyperparameter experiments."""
        
        # Define experiment configurations
        experiments = [
            {
                'name': 'baseline',
                'word_emb_dim': 100,
                'pos_emb_dim': 32,
                'lstm_dim': 256,
                'lr': 2e-3,
                'optimizer': 'adam',
                'weight_decay': 0.0
            },
            {
                'name': 'larger_embeddings',
                'word_emb_dim': 200,
                'pos_emb_dim': 64,
                'lstm_dim': 256,
                'lr': 2e-3,
                'optimizer': 'adam',
                'weight_decay': 0.0
            },
            {
                'name': 'larger_lstm',
                'word_emb_dim': 100,
                'pos_emb_dim': 32,
                'lstm_dim': 512,
                'lr': 2e-3,
                'optimizer': 'adam',
                'weight_decay': 0.0
            },
            {
                'name': 'lower_lr',
                'word_emb_dim': 100,
                'pos_emb_dim': 32,
                'lstm_dim': 256,
                'lr': 1e-3,
                'optimizer': 'adam',
                'weight_decay': 0.0
            },
            {
                'name': 'higher_lr',
                'word_emb_dim': 100,
                'pos_emb_dim': 32,
                'lstm_dim': 256,
                'lr': 5e-3,
                'optimizer': 'adam',
                'weight_decay': 0.0
            },
            {
                'name': 'sgd_optimizer',
                'word_emb_dim': 100,
                'pos_emb_dim': 32,
                'lstm_dim': 256,
                'lr': 0.1,
                'optimizer': 'sgd',
                'weight_decay': 0.0
            },
            {
                'name': 'with_weight_decay',
                'word_emb_dim': 100,
                'pos_emb_dim': 32,
                'lstm_dim': 256,
                'lr': 2e-3,
                'optimizer': 'adam',
                'weight_decay': 1e-4
            }
        ]
        
        print(f"Running {len(experiments)} experiments...")
        
        for config in experiments:
            try:
                result = self.train_model(config, max_epochs=5)  # Reduced epochs for faster experimentation
                self.results.append(result)
                print(f"✓ Completed {config['name']}: LAS = {result['best_las']:.2f}")
            except Exception as e:
                print(f"✗ Failed {config['name']}: {e}")
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save experiment results to JSON file."""
        # Convert tensors to lists for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = {
                'config': result['config'],
                'best_las': float(result['best_las']),
                'training_time': float(result['training_time']),
                'epochs': result['epochs']
            }
            serializable_results.append(serializable_result)
        
        with open('experiment_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to experiment_results.json")
    
    def print_summary(self):
        """Print a summary of all experiments."""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        # Sort by LAS
        sorted_results = sorted(self.results, key=lambda x: x['best_las'], reverse=True)
        
        print(f"{'Rank':<5} {'Config':<20} {'LAS':<8} {'Time (s)':<10}")
        print("-" * 50)
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i:<5} {result['config']['name']:<20} {result['best_las']:<8.2f} {result['training_time']:<10.1f}")
        
        best_result = sorted_results[0]
        print(f"\n🏆 Best Configuration: {best_result['config']['name']}")
        print(f"   LAS: {best_result['best_las']:.2f}")
        print(f"   Training Time: {best_result['training_time']:.1f} seconds")
        print(f"   Config: {best_result['config']}")

def main():
    """Main function to run experiments."""
    print("Starting Hyperparameter Experiments")
    print("="*50)
    
    runner = ExperimentRunner()
    runner.run_experiments()

if __name__ == "__main__":
    main() 