# SmolVLA Source Inspection

## Prefix

The actual SmolVLA implementation constructs a dynamic prefix:

[IMAGE][LANGUAGE][STATE]

VLIA extends it to:

[IMAGE][LANGUAGE][INTENTION][STATE]

Therefore:

L_prefix -> L_prefix + 1

No fixed prefix length is assumed.

## Dimensions

The currently loaded SmolVLA backbone uses a VLM hidden dimension of 960.

This value is a property of the current backbone and is not hard-coded into
the VLIA architecture.

The current VLIA intention latent dimension is 256.

## Attention

Observed prefix attention behavior:

- IMAGE: segment 0
- LANGUAGE: segment 0
- STATE: segment 1

VLIA V0 uses:

- INTENTION: segment 1
- STATE: segment 1

The intention padding mask is enabled.

## Training Path

SmolVLA uses flow matching for action generation.

VLIA preserves the original action objective and modifies only the prefix
context by adding the intention representation.

Backpropagation from the action loss to the IntentionAdapter has been verified.

## Inference Path

The prefix is processed before the inference KV cache is constructed.

Because the intention token is inserted into the prefix before this step, the
intention representation becomes part of the cached context used during action
generation.

## Verified

- Dynamic prefix verified.
- Intention insertion position verified.
- Prefix masks verified.
- Real model forward verified.
- Gradient to IntentionAdapter verified.
- Action sampling verified.
- KV-cache path verified.
- Policy-level training verified.
- Policy-level inference verified.
- No-intention fallback verified.
- Single-load policy initialization verified.
