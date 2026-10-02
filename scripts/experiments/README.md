# Experimental scripts

This directory contains exploratory Stage-A models and ablations retained for
research provenance and reproducibility.

They are not all part of the selected VLIA pipeline.

Current selected direction:

visual representation
    -> predicted WHAT
    -> WHAT-guided WHY reasoning
    -> residual reranker trained with frozen base
    -> z_int

The current downstream trainer is:

scripts/train_matched_oracle_LIBERO7D_FIXED.py

See the repository README and docs/ROADMAP.md for the active experimental path.
