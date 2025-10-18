import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        # Using nn.ModuleDict to easily access layers by name for hooks
        self.layers = nn.ModuleDict({
            'fc1': nn.Linear(vocab_size, hidden_size),
            'act1': nn.GELU(),
            'fc2': nn.Linear(hidden_size, hidden_size),
            'act2': nn.GELU(),
            'fc3': nn.Linear(hidden_size, vocab_size)
        })

    def forward(self, x):
        x = self.layers['act1'](self.layers['fc1'](x))
        x = self.layers['act2'](self.layers['fc2'](x))
        x = self.layers['fc3'](x)
        return x

# --- Activation Hooking ---
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

# --- Synthetic Data ---
def get_data(seq_len=16, vocab_size=16, batch_size=128):
    """Generates batches of data for a simple algorithmic task."""
    # Task: predict the next token in a sequence, which is (current_token + 1) % vocab_size
    while True:
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        inputs = F.one_hot(tokens, num_classes=vocab_size).float()
        labels = (tokens + 1) % vocab_size
        yield inputs, labels

# --- Main Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Train a small MLP for mechanistic analysis.")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw", help="Optimizer to use.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps.")
    parser.add_argument("--num_steps", type=int, default=1001, help="Total number of training steps.")
    parser.add_argument("--vocab_size", type=int, default=16, help="Vocabulary size.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup model, optimizer, and loss
    model = MLP(vocab_size=args.vocab_size, hidden_size=args.hidden_size)
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    criterion = nn.CrossEntropyLoss()
    data_generator = get_data(vocab_size=args.vocab_size, batch_size=args.batch_size)

    # Register forward hooks to capture activations
    # We are interested in the output of activation functions
    hook_handles = []
    for name, layer in model.layers.items():
        if isinstance(layer, nn.GELU):
            handle = layer.register_forward_hook(get_activation(name))
            hook_handles.append(handle)

    # --- Training ---
    for step in tqdm(range(args.num_steps)):
        inputs, labels = next(data_generator)
        
        logits = model(inputs)
        loss = criterion(logits.view(-1, args.vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        
        # Calculate gradient norm
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        optimizer.step()

        if step % args.save_every == 0:
            # --- Logging and Checkpointing ---
            
            # Calculate metrics
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == labels).float().mean().item()
            
            total_param_norm = 0.0
            for p in model.parameters():
                param_norm = p.data.norm(2)
                total_param_norm += param_norm.item() ** 2
            total_param_norm = total_param_norm ** 0.5
            
            print(f"\n--- Step {step} ---")
            print(f"Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")
            print(f"Gradient Norm: {total_grad_norm:.4f} | Parameter Norm: {total_param_norm:.4f}")

            # Save checkpoint
            checkpoint_dir = f"checkpoints/{args.optimizer}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            # Save activations
            activation_dir = f"analysis/activations/{args.optimizer}"
            os.makedirs(activation_dir, exist_ok=True)
            activation_path = os.path.join(activation_dir, f"step_{step}.npz")
            np.savez(activation_path, **activations)
            print(f"Saved activations to {activation_path}")
            print("------------------")

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()

    print("Training complete.")

if __name__ == "__main__":
    main()