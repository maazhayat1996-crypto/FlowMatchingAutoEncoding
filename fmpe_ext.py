"""
Small extensions of sbi 0.27's FMPE for long, interruptible cluster runs.

ChunkedFMPE adds three things on top of `sbi.inference.FMPE`, none of which
change the loss, the optimiser or sbi's early-stopping rule:

1. `train_chunk(n)` trains for at most n more epochs and returns, so the caller
   can checkpoint and check a wall-time budget.  It works by making `_converged`
   report True at the chunk boundary.  The obvious alternative (calling
   `train(resume_training=True)` with a cumulative `max_num_epochs`) is not
   equivalent to uninterrupted training in sbi 0.27: when the loop stops on
   `max_num_epochs`, `_run_training_loop` reloads the best-validation weights,
   so every chunk boundary would roll the network back.  `_converged` results
   are cached per epoch so a boundary epoch is not counted twice.

2. `save_checkpoint` / `load_checkpoint` pickle the inference object (network,
   Adam state, train/val split, epoch counter, early-stopping state) *without*
   the training data, which for 1e7 simulations would be ~1.4 GB per write.
   After loading, `restore_data` re-appends the same simulations; the split
   indices refer to them, so they must be regenerated from the same seed or
   loaded from the same cache.  sbi's own `__getstate__` already drops the
   TensorBoard tracker from the pickle; `set_tracker` re-attaches one.

3. An optional fast data loader (`fast_loader=True`).  sbi's default is a
   torch DataLoader over a TensorDataset with a SubsetRandomSampler, which
   indexes and collates one sample at a time.  `_TensorBatches` draws the same
   kind of batches (uniform random permutation of the training indices,
   drop_last) with one fancy-indexing op per tensor per batch, which matters
   at 1e7 samples.
"""

from __future__ import annotations

import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from sbi.inference import FMPE
from sbi.utils.tracking import TensorBoardTracker

# Attributes that hold the training data (dropped from checkpoints).
_DATA_ATTRS = ("_theta_roundwise", "_x_roundwise", "_prior_masks",
               "_data_round_index", "_proposal_roundwise")


class _TensorBatches:
    """
    Iterable of random mini-batches over `indices` of equally long tensors.

    Mimics what sbi's training loop needs from a DataLoader: iteration,
    `len()` (number of batches) and `.batch_size`.
    """

    def __init__(self, tensors: tuple[torch.Tensor, ...], indices: torch.Tensor,
                 batch_size: int):
        dev = tensors[0].device
        self.tensors = tuple(t.to(dev) for t in tensors)
        self.indices = indices.to(dev)
        self.batch_size = min(batch_size, len(indices))

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size

    def __iter__(self):
        perm = self.indices[torch.randperm(len(self.indices), device=self.indices.device)]
        bs = self.batch_size
        for b in range(len(self)):
            idx = perm[b * bs:(b + 1) * bs]
            yield tuple(t[idx] for t in self.tensors)


class ChunkedFMPE(FMPE):
    """FMPE that can train in chunks of epochs and checkpoint between them."""

    def __init__(self, *args: Any, fast_loader: bool = True, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.fast_loader = fast_loader
        self.sbi_converged = False        # sbi's early-stopping criterion fired
        self._chunk_end: int | None = None
        self._last_check: tuple[int, bool] | None = None
        self.train_pass_s: list[float] = []   # wall time of each epoch's training pass

    # ---- training in chunks ------------------------------------------------ #

    @property
    def epochs_trained(self) -> int:
        return int(getattr(self, "epoch", 0))

    @property
    def best_val_loss(self) -> float:
        return float(self._best_val_loss)

    def train_chunk(self, n_epochs: int, **train_kwargs: Any) -> bool:
        """
        Train for at most `n_epochs` more epochs.  Returns True when training is
        finished, i.e. sbi's early stopping fired or `max_num_epochs` was reached.
        """
        resume = self._neural_net is not None
        self._chunk_end = self.epochs_trained + n_epochs
        try:
            self.train(resume_training=resume, **train_kwargs)
        finally:
            self._chunk_end = None
        # sbi appends one entry per train() call and uses their *sum* as the
        # TensorBoard step offset; with cumulative epoch counts that offset would
        # overshoot, so keep only the latest entry.
        for k in ("epochs_trained", "best_validation_loss"):
            del self._summary[k][:-1]
        max_epochs = train_kwargs.get("max_num_epochs", 2**31 - 1)
        return self.sbi_converged or self.epochs_trained > max_epochs

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        if self._last_check is not None and self._last_check[0] == epoch:
            conv = self._last_check[1]
        else:
            conv = super()._converged(epoch, stop_after_epochs)
            self._last_check = (epoch, conv)
        self.sbi_converged = conv
        if conv:
            return True
        return self._chunk_end is not None and epoch >= self._chunk_end

    def _train_epoch(self, train_loader, clip_max_norm, loss_args):
        # Time the optimisation pass alone (no validation) for ms/step; the
        # per-batch loss.item() inside sbi's loop already synchronises CUDA.
        t0 = time.time()
        loss = super()._train_epoch(train_loader, clip_max_norm, loss_args)
        if not hasattr(self, "train_pass_s"):          # checkpoints from older versions
            self.train_pass_s = []
        self.train_pass_s.append(time.time() - t0)
        return loss

    def _report_convergence_at_end(self, epoch: int, max_num_epochs: int) -> None:
        # Stay quiet at chunk boundaries; sbi would announce "converged".
        if self.sbi_converged or epoch > max_num_epochs:
            super()._report_convergence_at_end(epoch, max_num_epochs)
            print(flush=True)

    # ---- data loading ------------------------------------------------------ #

    def get_dataloaders(self, starting_round: int = 0, training_batch_size: int = 200,
                        validation_fraction: float = 0.1, resume_training: bool = False,
                        dataloader_kwargs: dict | None = None):
        if not self.fast_loader or dataloader_kwargs:
            return super().get_dataloaders(starting_round, training_batch_size,
                                           validation_fraction, resume_training,
                                           dataloader_kwargs)
        if len(self._theta_roundwise) == 1 and starting_round == 0:
            data = (self._theta_roundwise[0], self._x_roundwise[0], self._prior_masks[0])
        else:
            data = self.get_simulations(starting_round)
        n = data[0].size(0)
        n_train = int((1 - validation_fraction) * n)
        if not resume_training:          # same split rule as sbi
            perm = torch.randperm(n)
            self.train_indices, self.val_indices = perm[:n_train], perm[n_train:]
        return (_TensorBatches(data, self.train_indices, training_batch_size),
                _TensorBatches(data, self.val_indices, training_batch_size))

    # ---- checkpointing ----------------------------------------------------- #

    def set_tracker(self, logdir: str | Path) -> None:
        """Log to TensorBoard in `logdir`, replacing (and removing) any default one."""
        old = getattr(self, "_tracker", None)
        if isinstance(old, TensorBoardTracker) and Path(old.log_dir).resolve() != Path(logdir).resolve():
            old._writer.close()
            if "sbi-logs" in Path(old.log_dir).parts:     # created by __setstate__
                shutil.rmtree(old.log_dir, ignore_errors=True)
        self._tracker = TensorBoardTracker(SummaryWriter(str(logdir)))

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        """Atomically write {"inference": self (no data), "torch_rng": ..., **extra}."""
        path = Path(path)
        saved = {k: getattr(self, k) for k in _DATA_ATTRS}
        for k in _DATA_ATTRS:
            setattr(self, k, [])
        state = {"inference": self, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 **extra}
        tmp = path.with_name(path.name + ".tmp")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")        # sbi warns on every pickle
                torch.save(state, tmp)
            os.replace(tmp, path)
        finally:
            for k, v in saved.items():
                setattr(self, k, v)

    def restore_data(self, theta: torch.Tensor, x: torch.Tensor,
                     data_device: str | None = None) -> None:
        """Re-attach the training data after `load_checkpoint`."""
        for k in _DATA_ATTRS:
            setattr(self, k, [])
        self.append_simulations(theta, x, data_device=data_device)
        n = self._theta_roundwise[0].size(0)
        n_split = len(self.train_indices) + len(self.val_indices)
        if n != n_split:
            raise RuntimeError(f"checkpoint was trained on {n_split} valid simulations, "
                               f"but {n} were restored; use the same seed / sims cache")


def load_checkpoint(path: str | Path) -> dict:
    """Inverse of `ChunkedFMPE.save_checkpoint`; restores the torch RNG state."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = torch.load(path, weights_only=False)
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
