# VLIA Architecture V0

## Base SmolVLA

[IMAGE][LANGUAGE][STATE] -> Action

## VLIA

[IMAGE][LANGUAGE][INTENTION][STATE] -> Action

VLIA inserts exactly one intention token:

L_prefix -> L_prefix + 1

The prefix length is dynamic and is not hard-coded.

## Intention Representation

z_int -> IntentionAdapter -> intention token -> SmolVLA

The current intention latent dimension is:

d_int = 256

The IntentionAdapter dynamically projects z_int into the hidden dimension of
the loaded VLM backbone.

## Oracle VLIA

Semantic Why
    -> Intention Encoder
    -> z_int
    -> IntentionAdapter
    -> SmolVLA
    -> Action

## Predicted VLIA

Egocentric Context + Language + History
    -> Intention Encoder
    -> z_int
    -> IntentionAdapter
    -> SmolVLA
    -> Action

## Design Principle

Intention represents the semantic purpose of the current behavioral phase,
while actions remain embodiment-specific.
