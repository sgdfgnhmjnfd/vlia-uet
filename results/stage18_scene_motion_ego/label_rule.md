# EgoIntent label rule (fixed before any accuracy)

Label = first verb of the WHAT sentence (`what` / `local_intent`) of each step:

1. lower-case, replace every non-letter by a space, split on spaces;
2. skip leading words in ['a', 'again', 'also', 'an', 'and', 'c', 'camera', 'first', 'he', 'i', 'just', 'next', 'now', 'person', 'she', 'still', 'the', 'then', 'they', 'to', 'user', 'wearer'] or adverbs in ['briefly', 'carefully', 'continuously', 'firmly', 'gently', 'quickly', 'repeatedly', 'slowly'];
3. lemma by inflection only (irregular table + -ies/-ied -> y, -ing/-ed with doubled consonant undone, -es after sh/ch/ss/x/z, -s); no merging by meaning (place != put);
4. if the next word is a particle in ['away', 'back', 'down', 'off', 'out', 'up'], append it (`pick up` -> `pick_up`); on/in/over are not merged (usually prepositions: `place on table`).
5. keep verbs with >= 30 steps and >= 8 videos, at most the 10 most frequent; other steps are dropped.

Deviation from the first draft (before any accuracy): skipping every word ending in -ly also skipped `apply`; replaced by the fixed adverb list above. The kept set did not change.

Result: K = 10 (place, pick_up, grasp, wipe, hold, position, adjust, remove, align, walk), 1001 of 2503 steps kept, largest class 0.255; distinct kept labels per video: mean 6.63, min 2, 27 videos.

WHY is not used: 2032 distinct sentences over 2503 steps. Mean step duration 2.51 s (median 1.77 s).
