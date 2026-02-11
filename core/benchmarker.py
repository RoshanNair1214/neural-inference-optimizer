#This file contains the testing suite that measures the latency difference between the two systems.

import time
import torch
import numpy as np

class LatencyBenchmarker:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def run_benchmark(self, input_seq, iterations=50):
        """
        Compares Latency between Naive (Model A) and Engineered (Model B).
        """
        # --- Test Naive Model ---
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model.forward_naive(input_seq)
        naive_latency = (time.time() - start_time) / iterations

        # --- Test Engineered Model (Compiled + Cached) ---
        # Note: In a real scenario, we'd use torch.compile(self.model) here
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                cache = None
                # Simulate step-by-step decoding
                for i in range(input_seq.size(1)):
                    current_input = input_seq[:, :i+1]
                    _, cache = self.model.forward_engineered(current_input, kv_cache=cache)
        eng_latency = (time.time() - start_time) / iterations

        return {
            "naive_latency": naive_latency,
            "eng_latency": eng_latency,
            "improvement": (naive_latency - eng_latency) / naive_latency * 100
        }

    def print_results(self, results):
        print("-" * 50)
        print("PHASE 4 BENCHMARK RESULTS")
        print("-" * 50)
        print(f"Naive Latency (Model A):      {results['naive_latency']:.4f}s")
        print(f"Engineered Latency (Model B): {results['eng_latency']:.4f}s")
        print(f"Performance Improvement:       {results['improvement']:.2f}%")
        print("-" * 50)
