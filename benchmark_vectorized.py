import torch
import time
from BAA.orchestrator import AgenticDTT

def benchmark_routing():
    device = torch.device("cpu") # Benchmarking on CPU for this environment
    print(f"Device: {device}")
    
    # 2 layers is enough for comparison
    model = AgenticDTT(state_dim=64, action_dim=16, n_layers=2).to(device)
    
    # 1. Test standard pass (now vectorized)
    states = torch.randn(8, 64, 64).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(2):
        _ = model(states)
    
    # Benchmark Vectorized Regular
    print("Benchmarking Vectorized Regular...")
    start = time.time()
    for _ in range(10):
        _ = model(states)
    regular_time = time.time() - start
    print(f"Regular (Full) Pass (10 iterations): {regular_time:.4f}s")
    
    # 2. Test Vectorized Top-K
    print("Enabling Top-K Sparsity...")
    model.set_sparsity(enabled=True, top_k=2)
    
    start = time.time()
    for _ in range(10):
        _ = model(states)
    topk_time = time.time() - start
    print(f"Top-K (k=2) Pass (10 iterations): {topk_time:.4f}s")

    # Safety/Correctness check
    model.eval()
    out = model(states)
    print(f"Output shape check: {out['action'].shape}")
    assert out['action'].shape == (8, 64, 16)
    print("✓ Output shape correct.")

if __name__ == "__main__":
    benchmark_routing()
