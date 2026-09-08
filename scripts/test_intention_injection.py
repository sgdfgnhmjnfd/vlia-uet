import torch

from policies.smolvla.modeling_smolvla import VLIAIntentionModule


def main():
    batch_size = 4
    intention_dim = 256
    vlm_dim = 960

    module = VLIAIntentionModule(
        intention_dim=intention_dim,
        vlm_dim=vlm_dim,
    )

    z_int = torch.randn(batch_size, intention_dim)

    token = module(z_int)

    print("z_int:", z_int.shape)
    print("intention token:", token.shape)

    assert token.shape == (batch_size, 1, vlm_dim)

    loss = token.square().mean()
    loss.backward()

    grad = module.adapter.proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()

    print("loss:", float(loss))
    print("grad norm:", float(grad.norm()))
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
