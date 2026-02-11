[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoshanNair1214/neural-inference-optimizer/blob/main/notebooks/Neural_Systems.ipynb)
# Neural Systems Inference Optimizer

**Optimizing Transformer Latency via KV-Caching and Graph Compilation.**

## Project Overview
This project focuses on the **Systems Engineering** side of Large Language Models. The goal was to reduce inference latency by transitioning from a naive auto-regressive decoding process (O(N^2) complexity) to an engineered, cached approach (O(N) complexity).

## Methodology
1. **KV-Caching:** Implemented a Key-Value cache to store previous hidden states, eliminating redundant matrix multiplications for existing tokens.
2. **Torch Compilation:** Utilized `torch.compile` to fuse kernels and optimize the execution graph for the GPU/CPU.
3. **Tokenization Engineering:** Built a custom mapping to visualize how raw bits are transformed into embeddings and eventually predictions.

## Tech Stack
* **Engine:** PyTorch
* **Optimization:** KV-Caching, Graph Compilation
* **Model Type:** Transformer Architecture (Inference-focused)
