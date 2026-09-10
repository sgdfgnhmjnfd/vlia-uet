from datasets.egointent_adapter import (
    build_egointent_alignment_samples,
)


LABEL = (
    "/media/dhqg/d1/datasets/egointent/"
    "indoor/art_studio/draw/step_label.json"
)


samples = build_egointent_alignment_samples(LABEL)

assert len(samples) > 0

with_video = 0

for sample in samples:
    assert sample.why.strip()
    assert sample.task == "draw"

    obs = sample.observation

    assert obs["start_time"] <= obs["obs_end_time"]

    if obs["video_path"] is not None:
        with_video += 1

    # Stage-A must not expose future supervision.
    serialized = repr(sample)

    assert "plausible_next_steps" not in serialized
    assert "observed_next_step" not in serialized


print("=== EGOINTENT ADAPTER ===")
print("samples:", len(samples))
print("with video:", with_video)

print()
print("FIRST SAMPLE")
print(samples[0])

print()
print("SECOND SAMPLE")
print(samples[1])

print()
print("EGOINTENT ADAPTER: PASS")