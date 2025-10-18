import argparse
import torch
import numpy as np
from train_small_model import MLP, get_data
import torch.nn.functional as F

# --- Patching Hook ---
# A global variable to store the activation from the source model
source_activation = None

def patch_hook(model, input, output):
    """A forward hook that replaces the output of a layer with a source activation."""
    global source_activation
    # The hook replaces the layer's output with the source activation
    return source_activation

# --- Patching Function ---
def run_patching_experiment(
    source_model_path,
    target_model_path,
    patch_layer_name,
    vocab_size=16,
    hidden_size=64,
    batch_size=128,
):
    """
    Performs a causal patching experiment by replacing a layer's output in a target
    model with the activations from a source model.
    """
    global source_activation

    # --- Setup ---
    # Load models
    source_model = MLP(vocab_size=vocab_size, hidden_size=hidden_size)
    source_model.load_state_dict(torch.load(source_model_path))
    source_model.eval()

    target_model = MLP(vocab_size=vocab_size, hidden_size=hidden_size)
    target_model.load_state_dict(torch.load(target_model_path))
    target_model.eval()

    # Get a batch of data
    data_generator = get_data(vocab_size=vocab_size, batch_size=batch_size)
    inputs, labels = next(data_generator)

    # --- Baseline Accuracy (No Patching) ---
    with torch.no_grad():
        original_logits = target_model(inputs)
        original_preds = torch.argmax(original_logits, dim=-1)
        baseline_accuracy = (original_preds == labels).float().mean().item()

    # --- Run with Patching ---
    # 1. Get the source activation
    source_hook_handle = None
    def get_source_activation_hook(model, input, output):
        global source_activation
        source_activation = output.detach()
    
    # Find the layer in the source model to get the activation from
    source_layer_to_hook = source_model.layers[patch_layer_name]
    source_hook_handle = source_layer_to_hook.register_forward_hook(get_source_activation_hook)

    with torch.no_grad():
        _ = source_model(inputs) # Run a forward pass to capture the activation
    
    source_hook_handle.remove() # Clean up the hook

    # 2. Register the patching hook on the target model
    target_layer_to_patch = target_model.layers[patch_layer_name]
    patch_handle = target_layer_to_patch.register_forward_hook(patch_hook)

    # 3. Run the patched forward pass
    with torch.no_grad():
        patched_logits = target_model(inputs)
        patched_preds = torch.argmax(patched_logits, dim=-1)
        patched_accuracy = (patched_preds == labels).float().mean().item()

    patch_handle.remove() # Clean up

    # --- Report Results ---
    print(f"--- Causal Patching Experiment ---")
    print(f"Source Model: {source_model_path}")
    print(f"Target Model: {target_model_path}")
    print(f"Patched Layer: '{patch_layer_name}'")
    print("------------------------------------")
    print(f"Baseline Target Accuracy: {baseline_accuracy:.4f}")
    print(f"Patched Target Accuracy:  {patched_accuracy:.4f}")
    print(f"Impact (Patched - Baseline): {patched_accuracy - baseline_accuracy:.4f}")
    print("------------------------------------")
    
    return baseline_accuracy, patched_accuracy

def main():
    parser = argparse.ArgumentParser(description="Run a causal patching experiment.")
    parser.add_argument("--source_model", required=True, help="Path to the source model checkpoint (.pt).")
    parser.add_argument("--target_model", required=True, help="Path to the target model checkpoint (.pt).")
    parser.add_argument("--patch_layer", required=True, help="Name of the layer to patch (e.g., 'act1', 'act2').")
    args = parser.parse_args()

    run_patching_experiment(
        source_model_path=args.source_model,
        target_model_path=args.target_model,
        patch_layer_name=args.patch_layer
    )

if __name__ == "__main__":
    main()