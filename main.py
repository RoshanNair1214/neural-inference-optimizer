import torch
from core.transformer_engine import KVCacheTransformer
from core.benchmarker import LatencyBenchmarker

def main():
    # 1. Configuration - Consistent with Phase 4 of the notebook
    vocab_size = 50257
    seq_length = 128  # Context window size for benchmarking
    
    # 2. Initialize Model and Benchmarker
    # Uses the engine logic extracted from Neural_Systems.ipynb
    model = KVCacheTransformer(vocab_size=vocab_size)
    benchmarker = LatencyBenchmarker(model)
    
    # 3. Prepare Dummy Data
    dummy_input = torch.randint(0, vocab_size, (1, seq_length))
    
    print("Initializing Neural Systems Optimization Benchmark...")
    print(f"Sequence Length: {seq_length} tokens")
    
    # 4. Run Benchmark
    # This executes both Naive (O(N^2)) and Engineered (O(N)) passes
    results = benchmarker.run_benchmark(dummy_input)
    
    # 5. Output Final Technical Metrics
    benchmarker.print_results(results)

if __name__ == "__main__":
    main()
