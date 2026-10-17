# Hand-detection gate

- Mean frame detection rate: 0.5819 (11652/20024)
- Minimum per-scene rate: 0.1026
- Verdict: **FAIL**
- Preregistered rule: PASS iff overall rate >= 0.60 and every scene >= 0.30.

## Per-scene rates

- `art_studio`: 0.6181 (445/720)
- `bedroom`: 0.6721 (1027/1528)
- `clothing_closet`: 0.8882 (739/832)
- `deck`: 0.1026 (87/848)
- `farm`: 0.3218 (551/1712)
- `garage`: 0.6184 (1588/2568)
- `hallway`: 0.8453 (683/808)
- `kitchen`: 0.7156 (2267/3168)
- `laundry_room`: 0.1533 (130/848)
- `living_room`: 0.6193 (981/1584)
- `space`: 0.2478 (224/904)
- `workshop`: 0.7620 (2798/3672)
- `yard`: 0.1587 (132/832)
The hand and matched-center branches are skipped in Stage 4. Hand regions remain display-only; Stage 4 uses full frames only.
