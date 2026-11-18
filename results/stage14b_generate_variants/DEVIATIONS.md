# Stage 14b Part C (Qwen variants): deviations and notes

1. **G3 not run.** The safetensors index of Qwen2.5-VL-7B-Instruct gives 16,584,333,312 bytes (15.45 GiB) of 16-bit weights, and the GPU (RTX 2000 Ada) has 16,380 MiB (16.00 GiB) in total. With the CUDA context, the activations of 8 images and the KV cache, the model cannot run on this GPU in fp16 or bf16.
   - The check was made from the index file before downloading, so the 16.6 GB of weights were not downloaded.
   - This was decided and written in the pre-registration before any Part C sentence existed.

2. **Scorer without the template.** The task says the Stage 14 scorer "with the template" and also "unchanged". The Stage 14 scorer (`analyze_libero`) encodes sentences and descriptions as plain text, so the unchanged scorer was kept. G0 therefore reproduces the Stage 14 numbers exactly.

3. **Run order.** The second version of the task (Parts A, B, C, D) arrived while Part C was already generating its calibration sentences, so Part C ran before Part B.
   - The Part C pre-registration was committed before any Part C sentence was generated.
   - Part B was pre-registered before any Part C result was seen.

4. **bf16 instead of fp16** for the 3B writer: the Stage 14 loader, unchanged.

5. **The answer prefix is not forced.** The prompts ask for "I want", and the share of sentences that start with it is reported in `accuracy_by_p.csv` (`starts_I_want`).
