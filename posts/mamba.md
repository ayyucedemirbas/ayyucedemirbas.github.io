# Mamba
The transformer architecture has a well-known problem, it scales quadratically. If you double the length of the text you feed into a transformer, the computational cost and memory requirements quadruple. This is why language models historically struggle with massive context windows like entire books or massive codebases. These models were also not good at translation tasks. For decades, we used Recurrent Neural Networks (RNNs) and State Space Models (SSMs). These models process data one step at a time, compressing history into a fixed-size memory state. Theoretically, their cost scales linearly. Double the text, double the cost. So why aren't we using RNNs? The answer is the architecture of modern hardware. Standard recurrent models are historically terrible at utilizing the parallel processing power of modern Graphics Processing Units (GPUs). They hit the memory wall.



Introduced in late 2023, the Mamba architecture fundamentally shifts how we approach deep learning. It doesn't just present a clever new mathematical formula; it presents math that was specifically reverse-engineered to exploit the physical layout of modern GPUs. In this article, we are going to cover the hardware-aware innovations that allow Mamba to achieve Transformer-level performance with linear scaling, detailing the exact hardware bottlenecks and how Mamba's custom engineering bypasses them.


## The Hardware
A modern GPU (like an NVIDIA A100 or H100) is a massive, complex, hierarchical structure. In this structure, the speed of your program depends on almost entirely by how far your data has to travel. There are two primary places data is stored on a GPU, and the disparity between them is staggering:


1. HBM (High Bandwidth Memory / DRAM - Global Memory): It can hold a massive amount of data (40GB to 80GB on an A100), storing your model's static weights and all the input text. While it has a high overall bandwidth (~1.55 TB/s on an A100), accessing it is incredibly far and slow in terms of latency. Every time the GPU's calculation cores need something from HBM, it takes hundreds of clock cycles (200-400+) for the data delivery to arrive. Furthermore, to read efficiently, data must be fetched in perfectly aligned 32-byte chunks (Memory Coalescing). If your code reads scattered data, the effective speed plummets.


2. SRAM (Shared Memory / L1 Cache): Located physically directly next to the calculation cores (the Tensor Cores/ALUs). It is ridiculously fast, pushing up to ~19 TB/s of aggregate bandwidth with a latency of barely 20-30 cycles. However, it is agonizingly small. A massive NVIDIA A100 GPU only has about 192 Kilobytes of SRAM per processing block.


## Sequential Memory Bottleneck

GPUs are designed to do thousands of math operations simultaneously. To keep a GPU fed and running at 100% capacity, your code must hit a specific target called the Ridge Point. The Ridge Point is a measure of Arithmetic Intensity: the ratio of math calculations performed for every single byte of data pulled from the slower HBM. On an NVIDIA A100, the Ridge Point is roughly 195 FLOPs/Byte. If your code does fewer than 195 calculations per byte moved, your multi-thousand-dollar GPU cores will sit idle, starving for data. Matrix multiplication (the core of the Transformer) easily clears this bar. You pull a chunk of data, and the GPU crunches on it thousands of times. But traditional recurrent models (like RNNs or naive SSMs) have terrible Arithmetic Intensity (often hovering around 0.5 FLOPs/Byte). Their core logic relies on sequential steps. You cannot process word 2 until you have finished processing word 1. If you run this naively on a GPU, you trigger a highly inefficient cycle known as the Sequential Memory Bottleneck:

* Fetch the memory state from HBM.
* Do one tiny calculation with the new word (low arithmetic intensity).
* Write the new memory state back to HBM.
* Wait hundreds of cycles. Repeat for the next word.

Because the GPU has to wait for these slow memory transfers on every single step, its massive calculation cores sit completely idle. The mighty A100 GPU is reduced to a crawl, crippled by its own memory bus.

## The Core of Mamba

Before we look at how Mamba fixes the hardware problem, we need to understand what makes it mathematically special. Mamba is built on the foundation of State Space Models (SSMs). At a high level, an SSM takes a 1D signal (like a sequence of words) and projects it into a high-dimensional mathematical space. As new words come in, this state evolves, acting as a compressed memory buffer. The system is governed by three conceptual matrices:


* Matrix A (The Physics Engine): This dictates how the memory state naturally evolves or decays over time.
* Matrix B (The Input Projector): This determines how a new input word influences the memory state.
* Matrix C (The Output Projector): This reads the high-dimensional memory state and squashes it back down to a useful output prediction.


Because computers operate in discrete, digital steps (Token 1, Token 2) rather than continuous time, this continuous physics-like system must be translated into digital steps. Mamba does this using a rule called the Zero-Order Hold (ZOH). It assumes that an input signal (a word) remains perfectly constant until the next word arrives.

During this digital translation, a crucial parameter emerges: Delta (Δ).  Δ represents the step size or the duration of time between inputs.


## The Selection Mechanism

Older SSMs were completely rigid (Linear Time-Invariant, or LTI). They applied the exact same matrices (A, B, C) and the exact same Δ to every single word. If the model read a crucial piece of information, and then read 50 words of useless filler, the rigid updating rules would steadily decay and blur out the crucial information.
Mamba introduces Selection. Mamba makes the core parameters, specifically matrices B, C, and the Δ step-size, dependent on the input data itself.


Δ acts like a data-driven gatekeeper:

* The Forget/Ignore Gate: If Mamba sees a useless filler word ("um", "the", "and"), the neural network projects a Δ value close to zero. The model essentially says, "Ignore this input, lock the memory vault, and perfectly preserve what we already know."
* The Write Gate: If Mamba sees a highly relevant keyword, it cranks Δ up. A large Δ forces the historical memory to decay and writes the new input deeply into the state.
This solves the famous Induction Head problem, giving the model the explicit ability to selectively copy information from far back in its context window while ignoring the noise in between.


# The Catch: Breaking the Fast Path

By introducing this dynamic, input-dependent Selection mechanism, Mamba broke the LTI property. In older, rigid SSMs, you could use a math trick called a Fast Fourier Transform (FFT) to process the whole sequence at once in parallel like a giant convolution filter. Because Mamba's rules change on every single word based on what it reads, FFTs are mathematically impossible. Mamba must calculate the sequence step-by-step. Mamba is mathematically forced back into the Sequential Memory Bottleneck.

## The IO-Aware Engineering
The creators of the Mamba architecture realized that if they could write a custom GPU program (a CUDA kernel) that completely avoided excessive HBM reads and writes, they could make sequential processing fast. They achieved this using:

### 1-  Kernel Fusion
Instead of letting the default deep learning framework (like PyTorch) handle the memory, Mamba's creators wrote a custom fused kernel. Instead of moving data back and forth between the fast SRAM and the slower HBM, Mamba loads a large chunk of the input text directly into the high-speed SRAM. Once the data is in SRAM, the GPU performs the discretization (calculating the step sizes), the parameter expansion, and the sequential memory updates entirely in local, high-speed registers. The massive, expanded intermediate memory states are never written out to the slow HBM. They live and die in the microscopic fraction of a second they exist in the SRAM. Only the final, compressed output is sent back to HBM. By refusing to write intermediate steps to HBM, Mamba reduces memory bandwidth usage by a staggering factor of 16x.

### 2- Tiling and Channel Independence
How do you fit a sequence of 100000 words into an SRAM that only holds 192 Kilobytes? You use Tiling (Chunking) and Channel Independence. Unlike Transformers, which mix information across every single channel dimension of a word (the Attention cross-pollination), State Space Models treat every channel dimension as totally independent. Channel 1 doesn't care what Channel 200 is doing. Mamba splits the massive grid of data up:

1. It assigns a small group of channels to a specific GPU thread block.
2. It slices the long sequence into small temporal chunks (e.g., 1024 words at a time).
3. The block loads Chunk 1 into SRAM, processes it, saves the final memory state locally, loads Chunk 2, and continues.

It processes infinite sequence lengths with a fixed, tiny SRAM budget. Furthermore, Mamba uses cooperative loading to transpose the data as it moves from HBM to SRAM, ensuring that the GPU only performs coalesced (perfectly aligned) memory reads, preventing massive bandwidth penalties.

### 3- The Parallel Associative Scan

Even if the data is in fast memory, processing words strictly one after another (1, then 2, then 3...) leaves thousands of GPU parallel threads doing nothing. Mamba uses an algorithm called a Parallel Associative Scan (historically known as a Blelloch Prefix Sum). Imagine you need to add a sequence of numbers: 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8.

* A sequential process does this: (((1+2)+3)+4)... taking 7 sequential steps.
* A parallel scan groups them: Thread A adds 1+2. Thread B adds 3+4. Thread C adds 5+6. In the next fraction of a second, Thread A adds the results of the first two groups together.


Mamba maps this mathematical tree structure directly to the hardware. For short sequences (under 32 steps), it uses Warp Shuffles, a hardware feature allowing threads to read each other's registers directly with near-zero latency. For longer sequences, it coordinates using the SRAM. By turning a straight line of processing into a tree-like hierarchy, Mamba squashes the latency of processing a long sequence from an O(L) linear time constraint down to an O(log L) logarithmic time constraint.


### 4: The Backpropagation
Training an AI model is vastly harder than running one because of backpropagation. To update its weights and learn, the model must remember exactly what happened during the forward pass. Normally, models save all their intermediate math to HBM to use later during the backward learning step. Mamba’s internal memory state operates at an expansion factor (usually N=16). This means the internal state is 16 times larger than the standard model dimension. If you calculate the memory required for a naive SSM to save its state history for a single 8,192-token sequence on a single layer, it is about 512 Megabytes. Multiply that by a standard 32-layer model, and a modest batch size of 8. You suddenly need 128 Gigabytes of VRAM just to store the memory states. A standard 80GB NVIDIA A100 immediately crashes with an Out Of Memory (OOM) error. Mamba solves this with Selective Recomputation. During the forward pass, Mamba simply throws the massive expanded memory states away. It doesn't save them at all. When it comes time to learn (the backward pass), Mamba just grabs the tiny, compressed input data again, and re-calculates the massive expanded states on the fly in the fast SRAM, throwing them away the second the gradients are calculated.


You may think, isn't doing the math twice slower? No. Because of the Memory Wall. Reading a massive 128GB file from the slow HBM takes an eternity in GPU time. Re-doing the lightweight math locally in the ultra-fast SRAM is literally faster than waiting for the HBM to fetch the saved file. Mamba gets to keep its memory usage totally flat O(L), allowing it to train on sequences of hundreds of thousands of words without paying any wall-clock time penalty. It acts as a Bandwidth Amplifier: keeping data compressed in HBM, expanding it 16x only while inside the fast SRAM, doing the work, and compressing it back before sending it out.

### 5: Mamba vs. Transformers

**During Training:**

* Transformers scale quadratically (O(L^2) computation). A 32000-word context window requires an astronomical amount of compute compared to a 2000-word window, largely due to the every-token-looks-at-every-other-token nature of Attention.

* Mamba scales linearly (O(L) computation). Because of the hardware-aware scan and recomputation, doubling the sequence length just doubles the time. When sequences exceed about 2048 words, Mamba surpasses Transformers in efficiency.

**During Inference (Generation):**

* Transformers suffer from the KV Cache (Key-Value Cache) problem. In a Transformer, the Attention mechanism works by comparing the current word (the Query) against every single previous word to figure out what context matters. To avoid recalculating the mathematical representations of all previous words every time a new word is generated, the model saves their Keys and Values in a massive memory bank called the KV Cache. To generate word 10001, it must keep the cached representations of the previous 10000 words in the GPU's memory. This cache grows infinitely with every new word, eventually choking the GPU, draining bandwidth, and severely limiting how many users can ping the model at once.

* Mamba has a fixed state. Whether it has read 10 words or 1000000 words, the size of its memory state is exactly the same (O(1) memory). It requires zero KV cache. This means inference is incredibly fast, throughput is massive, and allows for massive batch sizes (serving many users at once).
