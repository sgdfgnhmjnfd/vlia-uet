# VLIA Experiment Contract

## Required Models

### SmolVLA Baseline

Vision + Language + State -> Action

### Oracle VLIA

Vision + Language + Oracle Intention + State -> Action

### Predicted VLIA

Egocentric Context
    -> Intention Encoder
    -> Predicted Intention
    -> SmolVLA
    -> Action

## Required Experimental Order

Baseline -> Oracle VLIA -> Predicted VLIA

Oracle VLIA must be evaluated before Predicted VLIA to separate the usefulness
of intention information from the difficulty of intention prediction.

## Controls

If resources permit:

- Random intention token
- Static task token

## Main Metric

The primary downstream metric is task success rate.

All main models must use the same evaluation protocol.

## Intention Metrics

Planned independent intention metrics include:

- Top-1 accuracy
- Top-3 accuracy
- Mean Reciprocal Rank
- Cosine similarity or cosine margin
- Phase separability
