# Spec: DFlash Block Diffusion Speculative Decoding POC for LlamaGen-XXL

## 1. Context & Objective
We are building a Proof of Concept (POC) to implement **DFlash (Block Diffusion Speculative Decoding)** on **LlamaGen-XXL**. 

LlamaGen converts a 256x256 image into a flat sequence of discrete visual tokens (256 tokens total) and generates them autoregressively, creating a severe memory-bandwidth bottleneck. We are going to accelerate this using parallel block-diffusion drafting based on the DFlash paper.

The goal is to implement an optimized proof of concept of this architecture, execute it on an **RTX 5080 (16GB VRAM)**, profile wall-clock latency, and verify that the accelerated output remains mathematically identical to the baseline.

---

## 2. Environment Status & Available Repositories
You have full access to the official codebases for both projects. They are already cloned into your workspace in the following directories:
* **./LlamaGen/** - Contains the complete visual autoregressive generation repository, model definitions, tokenizers, and weights.
* **./dflash/** - Contains the reference implementation of the text-based Block Diffusion Speculative Decoding framework.

**Your Architecture Mandate:**
You are completely untethered to rigid function definitions. You are expected and encouraged to **freely change, adapt, hack, and design any hybrid architecture** between these two repositories that you think will maximize performance, execution speed, and token acceptance rate for this visual POC.

---

## 3. Core Research Literature
Before writing any code, please inspect and digest the core concepts from these frameworks within your workspace and via literature:
1.  **DFlash Core Architecture:** *DFlash: Block Diffusion for Flash Speculative Decoding* (arXiv:2602.06036). Focus on how target hidden states are used to guide parallel block drafting.
2.  **Target Model Mechanics:** *Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation* (arXiv:2406.06525). Focus on how images are flattened into 1D visual token sequences.
3.  **Speculative Core Math:** *Fast Inference from Transformers via Speculative Decoding* (arXiv:2211.17192). Reference for the lossless verification loop.

---

## 4. High-Level Architecture Instructions
Please create a standalone execution script named `dflash_visual_poc.py` in your workspace. You must implement the following three logical pillars using any optimal design choices you see fit:

### Component 1: Hidden State & KV Injection Plumbing
* Analyze `LlamaGen/autoregressive/models/gpt.py` to map out its transformer layer configuration.
* Implement a system (e.g., via PyTorch forward hooks) to capture intermediate hidden layer tensors from the `LlamaGen-XXL` target model during execution. 
* Ensure these extracted features can be cleanly mapped to guide the draft mechanism without breaking the main PyTorch computation graph.

### Component 2: The Block-Diffusion Drafter
* Build a mock or ultra-lightweight drafting model class.
* The drafter must take the current token context and your extracted target features, and output a spatial block of B predicted visual tokens simultaneously (default B=16).
* Incorporate a parameter to introduce a controlled error rate (e.g., 10% noise) to simulate the performance of a real, imperfect diffusion drafter.
* *Optimization Constraint:* To accurately measure verification engine gains, treat the mock drafting time overhead mathematically as O(1) during speedup calculations.

### Component 3: Parallel Visual Verification Engine
* Implement the core speculative decoding engine. 
* Take the block of B drafted tokens, append them to the current image token matrix, and evaluate them through `LlamaGen-XXL` in **a single parallel forward pass**.
* Greedily accept correct tokens up to the first mismatch, apply the target model's correction, and shift the generation window forward.

---

## 5. Benchmarking & Validation Rules
Your script must execute a strict, controlled test:

1.  **Hardware & Precision:** Load `LlamaGen-XXL` and the VQ tokenizer in `torch.bfloat16` to optimize execution on the RTX 5080.
2.  **The Comparison:** Generate a 256-token image sequence using LlamaGen's default sequential autoregressive loop. Then, generate the exact same image sequence using your newly designed speculative block decoding engine.
3.  **Strict Correctness:** Implement a `torch.equal()` check to ensure the final output tokens match perfectly under a 0% draft error rate condition, proving your engine is completely lossless.
4.  **Reporting:** Print a clean Markdown table summarizing:
    * Sequential Baseline Latency (Seconds)
    * DFlash Visual Latency (Seconds)
    * Net Acceleration Factor (x-fold speedup)
    * Average Token Acceptance Rate per Block (%)

Please review both `./LlamaGen/` and `./dflash/`, design the most effective visual adaptation possible, and run the benchmark.