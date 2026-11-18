# Stage 14b: accuracy by prefix (299 units, nearest of the 10 descriptions, strict top-1)

Chosen on calibration: G = G2, G+obj = O1. G+obj is given the scene's object names and is not open-ended.

| variant | p=0 | p=10 | p=25 | p=50 | p=75 | p=100 |
|---|---|---|---|---|---|---|
| G0 (Stage 14) | 33.8 [30.8; 36.8] | 28.8 [26.1; 31.4] | 29.4 [26.8; 32.1] | 36.5 [33.1; 39.8] | 41.8 [38.5; 45.2] | 41.8 [38.8; 44.8] |
| G2 (chosen G) | 35.8 [32.4; 39.1] | 27.8 [24.7; 30.8] | 30.8 [27.4; 34.1] | 39.8 [36.5; 43.1] | 42.1 [38.8; 45.5] | 42.8 [39.5; 46.2] |
| O1 (G+obj) | 40.8 [38.1; 43.5] | 37.8 [35.1; 40.1] | 44.8 [42.5; 47.2] | 44.5 [41.8; 46.8] | 41.8 [38.8; 44.8] | 43.8 [40.8; 46.8] |
| G0 blank | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| G2 blank | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| O1 blank | 50.2 | 50.2 | 50.2 | 50.2 | 50.2 | 50.2 |
| F (Stage 12) | 64.9 | 68.9 | 69.9 | 69.6 | – | 67.6 |
| C (Stage 15, 379 demos) | 88.7 | 96.3 | 99.7 | 100.0 | 100.0 | 100.0 |

| rule | value | 95% CI | verdict |
|---|---:|---|---|
| GB-1: hit(G2) - hit(G0) at p=25, paired CI lower bound > 0 | +0.013 | [-0.037, +0.064] | FAIL |
| GB-2: accuracy(G2) at p=50, CI lower bound > 30% | +0.398 | [+0.365, +0.431] | PASS |
| GB-3: hit(G2) - hit(G2-blank) at p=50, paired CI lower bound > 0 | +0.298 | [+0.227, +0.368] | PASS |

| descriptive | p | delta [95% CI] |
|---|---:|---|
| O1 - G2 | 0 | +0.050 [-0.020, +0.120] |
| O1 - G2 | 10 | +0.100 [+0.043, +0.157] |
| O1 - G2 | 25 | +0.140 [+0.087, +0.194] |
| O1 - G2 | 50 | +0.047 [-0.007, +0.100] |
| O1 - G2 | 75 | -0.003 [-0.060, +0.057] |
| O1 - G2 | 100 | +0.010 [-0.047, +0.067] |

Blank sentences: G0: ['I want the objects on the table to be organized and placed in their designated spots after completing the task.']; G2: ['I want to place the book on the table, followed by the pen and notebook next to it, then the pencil and eraser beside them, and finally the ruler and calculator on top of the book']; O1: ['I want the robot arm to move the black book from the desk caddy to the study table.', 'I want the robot arm to move the first moka pot from the kitchen table to the flat stove.', 'I want the robot arm to move the moka pot from the kitchen table to the flat stove.', 'I want the robot arm to move the red coffee mug from the living room table to the plate on the same table.', 'I want the robot arm to move the red coffee mug from the living room table to the white yellow mug on the same table.', 'I want the robot arm to move the white yellow mug from the kitchen table to the microwave.', 'I want the robot arm to move the wine bottle from the white cabinet to the wine rack on the kitchen table.', 'I want the robot arm to move towards the alphabet soup on the living room table, then pick it up and place it into the basket.', 'I want the robot arm to pick up the tomato sauce from the table and move it to the center of the table.']
