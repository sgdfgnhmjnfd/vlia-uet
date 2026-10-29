# Archived scripts

Historical experiment code, kept for provenance. Nothing here is used by the current pipeline:
no script outside `scripts/archive/` imports or calls these files (checked on 2026-10-29).

- Files were moved with `git mv`, so `git log --follow <new path>` shows their full history.
- Imports inside these files may need path fixes to run again, because they assumed `scripts/` as the working folder.
- Result notes under `results/` keep the old paths; use the table below to find each file.

Any trainer marked `BUGGY_6D` used an invalid 6-D LIBERO action schema. LIBERO experiments need 7-D actions.

## `stage_a_pilot/`

Stage 0 pilot on 8 videos: trained intention encoders, PCA bridges, overfit and alignment checks (thesis Stage 0). Also the former `scripts/experiments/stage_a/`.

| old path | new path |
|---|---|
| `scripts/cache_egointent_stage_a_features.py` | `scripts/archive/stage_a_pilot/cache_egointent_stage_a_features.py` |
| `scripts/cache_structured_text_targets.py` | `scripts/archive/stage_a_pilot/cache_structured_text_targets.py` |
| `scripts/download_egointent_pilot.py` | `scripts/archive/stage_a_pilot/download_egointent_pilot.py` |
| `scripts/eval_stage_a_cached.py` | `scripts/archive/stage_a_pilot/eval_stage_a_cached.py` |
| `scripts/export_stage_a_clean_v1.py` | `scripts/archive/stage_a_pilot/export_stage_a_clean_v1.py` |
| `scripts/fit_structured_pca.py` | `scripts/archive/stage_a_pilot/fit_structured_pca.py` |
| `scripts/sync_stage_a_why_to_temporal_cache.py` | `scripts/archive/stage_a_pilot/sync_stage_a_why_to_temporal_cache.py` |
| `scripts/test_architectures_quick.py` | `scripts/archive/stage_a_pilot/test_architectures_quick.py` |
| `scripts/test_egointent_adapter.py` | `scripts/archive/stage_a_pilot/test_egointent_adapter.py` |
| `scripts/test_egointent_split.py` | `scripts/archive/stage_a_pilot/test_egointent_split.py` |
| `scripts/test_egointent_stage_a_split.py` | `scripts/archive/stage_a_pilot/test_egointent_stage_a_split.py` |
| `scripts/test_egointent_video.py` | `scripts/archive/stage_a_pilot/test_egointent_video.py` |
| `scripts/test_intention_encoder_v0.py` | `scripts/archive/stage_a_pilot/test_intention_encoder_v0.py` |
| `scripts/test_intention_text_encoder.py` | `scripts/archive/stage_a_pilot/test_intention_text_encoder.py` |
| `scripts/test_intention_visual_encoder.py` | `scripts/archive/stage_a_pilot/test_intention_visual_encoder.py` |
| `scripts/test_stage_a_alignment.py` | `scripts/archive/stage_a_pilot/test_stage_a_alignment.py` |
| `scripts/test_stage_a_fixed_pca_bridge.py` | `scripts/archive/stage_a_pilot/test_stage_a_fixed_pca_bridge.py` |
| `scripts/test_stage_a_overfit.py` | `scripts/archive/stage_a_pilot/test_stage_a_overfit.py` |
| `scripts/test_why_pca_256.py` | `scripts/archive/stage_a_pilot/test_why_pca_256.py` |
| `scripts/train_prefix_mean_delta_ablation.py` | `scripts/archive/stage_a_pilot/train_prefix_mean_delta_ablation.py` |
| `scripts/train_stage_a_ablation.py` | `scripts/archive/stage_a_pilot/train_stage_a_ablation.py` |
| `scripts/train_stage_a_cached.py` | `scripts/archive/stage_a_pilot/train_stage_a_cached.py` |
| `scripts/train_stage_a_clean_v1.py` | `scripts/archive/stage_a_pilot/train_stage_a_clean_v1.py` |
| `scripts/train_stage_a_contrastive.py` | `scripts/archive/stage_a_pilot/train_stage_a_contrastive.py` |
| `scripts/train_structured_temporal_intention.py` | `scripts/archive/stage_a_pilot/train_structured_temporal_intention.py` |
| `scripts/visualize_vlia_pilot.py` | `scripts/archive/stage_a_pilot/visualize_vlia_pilot.py` |
| `scripts/experiments/stage_a/*` | `scripts/archive/stage_a_pilot/experiments/*` |

## `temporal_encoders/`

Temporal encoders over cached frame features (attention, GRU, guided). None beat mean pooling.

| old path | new path |
|---|---|
| `scripts/audit_temporal_vs_static_cache.py` | `scripts/archive/temporal_encoders/audit_temporal_vs_static_cache.py` |
| `scripts/cache_egointent_temporal_features.py` | `scripts/archive/temporal_encoders/cache_egointent_temporal_features.py` |
| `scripts/enrich_egointent_temporal_cache.py` | `scripts/archive/temporal_encoders/enrich_egointent_temporal_cache.py` |
| `scripts/train_temporal_guided_v2.py` | `scripts/archive/temporal_encoders/train_temporal_guided_v2.py` |
| `scripts/train_temporal_intention.py` | `scripts/archive/temporal_encoders/train_temporal_intention.py` |

## `egointent_trained_heads/`

Trained WHAT/WHY heads, hard-negative, soft-MRR, distillation and reranker variants on EgoIntent, with their sweep scripts. Superseded by the training-free pipeline.

| old path | new path |
|---|---|
| `run_what_hardneg_sweep.sh` | `scripts/archive/egointent_trained_heads/run_what_hardneg_sweep.sh` |
| `scripts/analyze_egointent_why_errors.py` | `scripts/archive/egointent_trained_heads/analyze_egointent_why_errors.py` |
| `scripts/eval_hardneg_softmrr_fusion.py` | `scripts/archive/egointent_trained_heads/eval_hardneg_softmrr_fusion.py` |
| `scripts/run_egointent_late_fusion.py` | `scripts/archive/egointent_trained_heads/run_egointent_late_fusion.py` |
| `scripts/run_frozen_reranker_sweep.sh` | `scripts/archive/egointent_trained_heads/run_frozen_reranker_sweep.sh` |
| `scripts/run_jobA_remaining.sh` | `scripts/archive/egointent_trained_heads/run_jobA_remaining.sh` |
| `scripts/run_softmrr_weight_sweep.sh` | `scripts/archive/egointent_trained_heads/run_softmrr_weight_sweep.sh` |
| `scripts/train_egointent_confidence_gate.py` | `scripts/archive/egointent_trained_heads/train_egointent_confidence_gate.py` |
| `scripts/train_egointent_confidence_scalar.py` | `scripts/archive/egointent_trained_heads/train_egointent_confidence_scalar.py` |
| `scripts/train_egointent_dualhead.py` | `scripts/archive/egointent_trained_heads/train_egointent_dualhead.py` |
| `scripts/train_egointent_frozen_oracle_distill.py` | `scripts/archive/egointent_trained_heads/train_egointent_frozen_oracle_distill.py` |
| `scripts/train_egointent_frozen_reranker.py` | `scripts/archive/egointent_trained_heads/train_egointent_frozen_reranker.py` |
| `scripts/train_egointent_full_baselines.py` | `scripts/archive/egointent_trained_heads/train_egointent_full_baselines.py` |
| `scripts/train_egointent_full_intention.py` | `scripts/archive/egointent_trained_heads/train_egointent_full_intention.py` |
| `scripts/train_egointent_guided.py` | `scripts/archive/egointent_trained_heads/train_egointent_guided.py` |
| `scripts/train_egointent_hierarchical_fixed.py` | `scripts/archive/egointent_trained_heads/train_egointent_hierarchical_fixed.py` |
| `scripts/train_egointent_oracle_diagnostic.py` | `scripts/archive/egointent_trained_heads/train_egointent_oracle_diagnostic.py` |
| `scripts/train_egointent_reasoner_distill.py` | `scripts/archive/egointent_trained_heads/train_egointent_reasoner_distill.py` |
| `scripts/train_egointent_residual_distill.py` | `scripts/archive/egointent_trained_heads/train_egointent_residual_distill.py` |
| `scripts/train_egointent_softmrr.py` | `scripts/archive/egointent_trained_heads/train_egointent_softmrr.py` |
| `scripts/train_egointent_strong_what.py` | `scripts/archive/egointent_trained_heads/train_egointent_strong_what.py` |
| `scripts/train_egointent_what_hardneg.py` | `scripts/archive/egointent_trained_heads/train_egointent_what_hardneg.py` |
| `scripts/train_egointent_what_mixture.py` | `scripts/archive/egointent_trained_heads/train_egointent_what_mixture.py` |
| `scripts/train_egointent_why_hardneg.py` | `scripts/archive/egointent_trained_heads/train_egointent_why_hardneg.py` |
| `scripts/train_egointent_why_hardneg_fixed.py` | `scripts/archive/egointent_trained_heads/train_egointent_why_hardneg_fixed.py` |

## `crossview_enigma/`

Ego-exo pretraining on ENIGMA-360. It hurt WHY retrieval.

| old path | new path |
|---|---|
| `scripts/cache_enigma360_crossview_features.py` | `scripts/archive/crossview_enigma/cache_enigma360_crossview_features.py` |
| `scripts/eval_enigma_crossview_action_retrieval.py` | `scripts/archive/crossview_enigma/eval_enigma_crossview_action_retrieval.py` |
| `scripts/eval_enigma_crossview_retrieval.py` | `scripts/archive/crossview_enigma/eval_enigma_crossview_retrieval.py` |
| `scripts/eval_enigma_crossview_temporal_matching.py` | `scripts/archive/crossview_enigma/eval_enigma_crossview_temporal_matching.py` |
| `scripts/train_egointent_why_crossview_twohead.py` | `scripts/archive/crossview_enigma/train_egointent_why_crossview_twohead.py` |
| `scripts/train_egointent_why_plus_crossview.py` | `scripts/archive/crossview_enigma/train_egointent_why_plus_crossview.py` |
| `scripts/train_enigma_crossview_projection.py` | `scripts/archive/crossview_enigma/train_enigma_crossview_projection.py` |
| `scripts/train_enigma_pretrain_then_why.py` | `scripts/archive/crossview_enigma/train_enigma_pretrain_then_why.py` |
| `scripts/validate_crossview_dataset.py` | `scripts/archive/crossview_enigma/validate_crossview_dataset.py` |

## `ego4d_pilots/`

Early Ego4D / HF Ego4D pilots, the intention-labeler versions and INSIGHT adapters.

| old path | new path |
|---|---|
| `scripts/audit_intention_v12.py` | `scripts/archive/ego4d_pilots/audit_intention_v12.py` |
| `scripts/audit_phase_taxonomy.py` | `scripts/archive/ego4d_pilots/audit_phase_taxonomy.py` |
| `scripts/build_ego4d_pilot.py` | `scripts/archive/ego4d_pilots/build_ego4d_pilot.py` |
| `scripts/build_hf_ego4d_intention_pilot.py` | `scripts/archive/ego4d_pilots/build_hf_ego4d_intention_pilot.py` |
| `scripts/build_hf_ego4d_pilot.py` | `scripts/archive/ego4d_pilots/build_hf_ego4d_pilot.py` |
| `scripts/export_intention_review_batch.py` | `scripts/archive/ego4d_pilots/export_intention_review_batch.py` |
| `scripts/test_ego4d_lta_adapter.py` | `scripts/archive/ego4d_pilots/test_ego4d_lta_adapter.py` |
| `scripts/test_hf_ego4d_boundaries.py` | `scripts/archive/ego4d_pilots/test_hf_ego4d_boundaries.py` |
| `scripts/test_insight_adapter.py` | `scripts/archive/ego4d_pilots/test_insight_adapter.py` |
| `scripts/test_intention_alignment_dataset.py` | `scripts/archive/ego4d_pilots/test_intention_alignment_dataset.py` |
| `scripts/test_intention_alignment_pilot.py` | `scripts/archive/ego4d_pilots/test_intention_alignment_pilot.py` |
| `scripts/test_intention_labeler.py` | `scripts/archive/ego4d_pilots/test_intention_labeler.py` |
| `scripts/test_intention_labeler_v1.py` | `scripts/archive/ego4d_pilots/test_intention_labeler_v1.py` |
| `scripts/test_intention_labeler_v12.py` | `scripts/archive/ego4d_pilots/test_intention_labeler_v12.py` |

## `early_vlia/`

First SmolVLA baseline / text-intention / intention-token trainers, including the 6-D action-schema bug era. The current LIBERO trainers are `train_matched_oracle_LIBERO7D_FIXED.py` and `train_intention_only_LIBERO7D.py`.

| old path | new path |
|---|---|
| `scripts/audit_oracle_action_chunk_alignment.py` | `scripts/archive/early_vlia/audit_oracle_action_chunk_alignment.py` |
| `scripts/eval_policy.py` | `scripts/archive/early_vlia/eval_policy.py` |
| `scripts/plot_training_history.py` | `scripts/archive/early_vlia/plot_training_history.py` |
| `scripts/train_baseline.py` | `scripts/archive/early_vlia/train_baseline.py` |
| `scripts/train_intention_token.py` | `scripts/archive/early_vlia/train_intention_token.py` |
| `scripts/train_matched_oracle.py` | `scripts/archive/early_vlia/train_matched_oracle.py` |
| `scripts/train_text_intention.py` | `scripts/archive/early_vlia/train_text_intention.py` |
