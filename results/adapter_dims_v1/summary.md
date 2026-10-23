# Intention Adapter dimension tests (prereg v2, Stage 3)

Implementation correctness only (CPU, fp32, randomly initialised SmolVLA with the 7-D LIBERO checkpoint architecture fields). Not scientific evidence.

| test | d=256 | d=768 | d=1152 |
|---|---|---|---|
| finite_loss | PASS | PASS | PASS |
| adapter_grad_nonzero | PASS | PASS | PASS |
| prefix_len_plus_1 | PASS | PASS | PASS |
| intention_before_state | PASS | PASS | PASS |
| in_kv_cache | PASS | PASS | PASS |
| flag_off_identical | PASS | PASS | PASS |
| checkpoint_roundtrip | PASS | PASS | PASS |

## Details

- d=256: {"loss": 2.3837387561798096, "adapter_grad_norm": 0.07747265696525574, "prefix_len": {"base": 77, "vlia": 78}, "kv_len": {"with_intention": 78, "without": 77}, "flag_off_max_abs_diff": {"flag_off": 0.0, "z_int_none": 0.0, "prefix": 0.0}, "roundtrip_max_abs_diff": 0.0}
- d=768: {"loss": 2.3828983306884766, "adapter_grad_norm": 0.1295718550682068, "prefix_len": {"base": 77, "vlia": 78}, "kv_len": {"with_intention": 78, "without": 77}, "flag_off_max_abs_diff": {"flag_off": 0.0, "z_int_none": 0.0, "prefix": 0.0}, "roundtrip_max_abs_diff": 0.0}
- d=1152: {"loss": 2.387240171432495, "adapter_grad_norm": 0.16490647196769714, "prefix_len": {"base": 77, "vlia": 78}, "kv_len": {"with_intention": 78, "without": 77}, "flag_off_max_abs_diff": {"flag_off": 0.0, "z_int_none": 0.0, "prefix": 0.0}, "roundtrip_max_abs_diff": 0.0}

Overall: ALL PASS
