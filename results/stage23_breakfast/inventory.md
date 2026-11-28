# Stage 23 inventory (Breakfast, MS-TCN I3D features)

- zip: /data/breakfast_raw/data.zip  size=30210005282
- videos (groundTruth files): 1712
- videos missing feature file: 0
- name rule: `P<pp>_<cam>_P<pp>_<dish>` split on '_'; person=field0 (==field2), cam=field1, dish=field3
  - P03_cam01_P03_cereals -> ('P03', 'cam01', 'cereals')
  - P03_cam01_P03_coffee -> ('P03', 'cam01', 'coffee')
  - P03_cam01_P03_friedegg -> ('P03', 'cam01', 'friedegg')
  - P03_cam01_P03_milk -> ('P03', 'cam01', 'milk')
  - P03_cam01_P03_salat -> ('P03', 'cam01', 'salat')
- people: 52 (P03..P54)
- cams: {'cam01': 433, 'stereo01': 304, 'webcam01': 365, 'webcam02': 338, 'cam02': 272}
- dishes (10): {'cereals': 184, 'coffee': 167, 'friedegg': 173, 'juice': 162, 'milk': 187, 'pancake': 157, 'salat': 163, 'sandwich': 169, 'scrambledegg': 166, 'tea': 184}
- recordings (person+dish): 503; videos per recording: {1: 46, 2: 78, 3: 113, 4: 159, 5: 107}

## Official splits
- split1: train 1460 vids / 39 people; test 252 vids / 13 people (P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13, P14, P15); overlap people = []; train+test = 1712
- split2: train 1261 vids / 39 people; test 451 vids / 13 people (P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28); overlap people = []; train+test = 1712
- split3: train 1279 vids / 39 people; test 433 vids / 13 people (P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41); overlap people = []; train+test = 1712
- split4: train 1136 vids / 39 people; test 576 vids / 13 people (P42, P43, P44, P45, P46, P47, P48, P49, P50, P51, P52, P53, P54); overlap people = []; train+test = 1712
- each video in test exactly once across 4 splits: True

## Features vs groundTruth (headers only)
- feature (dim,dtype) counts: {(2048, 'float32'): 1712}; stored as (2048, T)
- T_feat - T_gt: n_mismatch=0, min=0, max=0 (mismatch -> crop to shorter)
- video length (s, at 15 fps per MS-TCN): median 91.5, min 8.7, max 649.4

## Fine action labels
- mapping.txt: 48 labels, SIL present: True
- total segments (incl. SIL): 11656
- non-SIL actions appearing in >=2 dishes (any count, seg>=1s): ['add_saltnpepper', 'butter_pan', 'crack_egg', 'pour_milk', 'pour_oil', 'pour_sugar', 'put_egg2plate', 'spoon_sugar', 'take_bowl', 'take_cup', 'take_eggs', 'take_knife', 'take_plate']

## A* (rule: >=3 dishes, each kept dish >=20 segs from >=8 people, seg>=1s, top-8 by n segs)
- candidates passing: 6; kept: 6
- pour_milk: K=4 | cereals 184seg/52p, coffee 117seg/35p, milk 204seg/51p, pancake 176seg/49p
- crack_egg: K=3 | friedegg 173seg/51p, pancake 157seg/49p, scrambledegg 166seg/49p
- take_plate: K=6 | friedegg 92seg/27p, juice 33seg/10p, pancake 114seg/36p, salat 42seg/12p, sandwich 22seg/8p, scrambledegg 124seg/35p
- butter_pan: K=3 | friedegg 68seg/19p, pancake 73seg/20p, scrambledegg 84seg/21p
- take_cup: K=3 | coffee 65seg/20p, milk 62seg/16p, tea 85seg/23p
- pour_oil: K=3 | friedegg 57seg/20p, pancake 45seg/13p, scrambledegg 56seg/19p

## Gate D0: data readable=True, no split leak=True, |A*|=6>=3 -> PASS

(inventory took 2s)
