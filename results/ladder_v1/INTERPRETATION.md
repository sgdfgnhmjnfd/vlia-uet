# Nested pipeline ladder status

Stage 2 stopped before any new encoder extraction, as required by the preregistration, because the cached SigLIP baseline did not reproduce the E1 K=5 accuracies to `1e-9` at k=2 and k=4. Each discrepancy is one query among 2503. The cache stores image features as float16; the original float32 feature matrix used for the E1 summary is not available to verify whether quantization caused the changed ranks.

No other encoder was loaded, no embedding extraction was started, and no nested-selection, descriptive-grid, control, N1, or N2 result was computed. Stages 3–5 are blocked by this baseline gate. The train/validation cache was used for the check; the test split was not read.
