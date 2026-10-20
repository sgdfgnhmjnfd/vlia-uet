# Selective E1 status

The preregistered baseline gate failed, so the selective analysis did not proceed and S1 has no verdict.

Using the cached SigLIP full-frame embeddings, the exact SigLIP text tokenizer settings from the E1 implementation, canonical `build_kway` candidates, and strict positive-over-all-distractors comparison:

- k=2: observed 693/2503 (0.276868), expected 0.277267.
- k=4: observed 690/2503 (0.275669), expected 0.275270.
- k=6: observed 703/2503 (0.280863), matching the expected value.
- k=8: observed 722/2503 (0.288454), matching the expected value.

The discrepancy at k=2 and k=4 is one query in each prefix. The cached image features are stored as float16; the original float32 image-feature arrays used to produce the baseline summary were not retained, so the exact baseline cannot be reconstructed from this cache. The discrepancy is consistent with quantization near a ranking boundary, but that cause cannot be verified from the saved artifacts.

No risk-coverage curves, AURC, bootstrap intervals, operational thresholds, or selective S1 verdict were computed. Only the train/validation manifest was used; the test split was not read.
