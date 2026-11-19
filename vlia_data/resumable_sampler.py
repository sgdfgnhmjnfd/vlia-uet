"""Resumable shuffled sampler for the 100k runs (added 2026-11-17; the 25k runs used DataLoader(shuffle=True)).

`ResumableRandomSampler` wraps torch's own `RandomSampler` with the DataLoader's generator. An uninterrupted run therefore
draws exactly the same random numbers in the same order as `DataLoader(shuffle=True, generator=g)`: the same batches at
every step (loss at step 1 = 3.584585, as at 25k).

On resume at step S (one batch per step, drop_last=True, `steps_per_epoch = len(loader)`):
  - the `S // steps_per_epoch` finished epochs are replayed on indices only: the inner sampler is exhausted once per
    epoch, which consumes the generator exactly as those epochs did;
  - in the current epoch the first `(S % steps_per_epoch) * batch_size` indices are skipped (no data is read).
The DataLoader draws its worker base seed from the same generator once, when its iterator is created, before the first
permutation, both in the original and in the resumed run, so the order of draws is identical.
"""
from __future__ import annotations

from torch.utils.data import RandomSampler, Sampler


class ResumableRandomSampler(Sampler[int]):
    def __init__(self, data_source, generator):
        self.inner = RandomSampler(data_source, generator=generator)
        self._skip_epochs = 0
        self._skip_items = 0

    def fast_forward(self, step: int, steps_per_epoch: int, batch_size: int) -> tuple[int, int]:
        """Prepare the next iteration to continue after `step` finished steps. Returns (finished epochs, skipped batches)."""
        self._skip_epochs, within = divmod(int(step), int(steps_per_epoch))
        self._skip_items = within * int(batch_size)
        return self._skip_epochs, within

    def __iter__(self):
        for _ in range(self._skip_epochs):
            for _ in self.inner:  # replay a finished epoch on indices only
                pass
        it = iter(self.inner)
        for _ in range(self._skip_items):
            next(it)
        self._skip_epochs = self._skip_items = 0
        yield from it

    def __len__(self) -> int:
        return len(self.inner)
