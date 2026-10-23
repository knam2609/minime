"""The training loop.

Everything here exists to make two runs comparable:

* tokens per optimiser step are fixed by the config, not by the device;
* validation uses the same fixed windows for every run -- on one set, or on
  several named ones (an adaptation run tracks general and personal text);
* the pre-clip gradient norm is recorded every step, because the optimiser
  experiment is partly an argument about gradient behaviour;
* throughput and MFU are recorded, so a wall-clock claim can be checked
  against whether the hardware was actually being used;
* a run can stop cleanly before its budget (a Kaggle session ends at 12 h)
  and resume later as the same run: same config, same precision, the loss
  scale included, every session recorded.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch

from ..config import Config
from ..data.mixture import MixtureSampler
from ..utils import costs as cost_utils
from ..utils.device import DeviceSpec, autocast_ctx, describe, make_scaler, resolve
from ..utils.seed import seed_everything
from ..utils.throughput import ThroughputMeter, format_duration
from .checkpoint import load_checkpoint, restore_checkpoint, rotate_checkpoints, save_checkpoint
from .logging import RunLogger
from .lr_sweep import flatten
from .optim import build_optimizer
from .schedule import apply_schedule, lr_at_step


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    history: list[dict] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        sampler: MixtureSampler,
        val_batches: Sequence[tuple[torch.Tensor, torch.Tensor]]
        | Mapping[str, Sequence[tuple[torch.Tensor, torch.Tensor]]]
        | None = None,
        run_dir: str | Path | None = None,
        experiment: str = "",
        venue: str = "local",
        wandb_project: str | None = None,
        costs_path: str | Path | None = None,
        data_info: dict | None = None,
        init_from: dict | None = None,
    ) -> None:
        self.cfg = config
        self.spec: DeviceSpec = resolve(config.training.device, config.training.precision)
        seed_everything(config.training.seed)

        self.model = model.to(self.spec.device)
        self.sampler = sampler
        # Named validation sets.  The first is the run's ``val_loss`` (best.pt,
        # the summary's final_val_loss); with more than one, every validation
        # also logs each set's loss under ``val_losses``.
        if isinstance(val_batches, Mapping):
            self.val_sets = {name: list(batches) for name, batches in val_batches.items()}
            empty = [name for name, batches in self.val_sets.items() if not batches]
            if empty:
                raise ValueError(f"validation set(s) without batches: {empty}")
        else:
            self.val_sets = {"val": list(val_batches)} if val_batches else {}
        self.optimizer = build_optimizer(self.model, config.training, self.spec.type)
        # Each group's peak comes from the config, not from a checkpoint's
        # param groups: those carry the LR of the step they were saved at.
        self.peak_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.scaler = make_scaler(self.spec)
        self.state = TrainState()

        self.run_dir = Path(run_dir or Path("experiments") / config.name)
        self.experiment = experiment or config.name
        self.venue = venue
        self.costs_path = Path(costs_path or Path("experiments") / "costs.csv")
        self.meter = ThroughputMeter()
        self._last_validation: tuple[int, dict[str, float]] | None = None
        self._wandb_project = wandb_project
        self.data_info = data_info
        # The checkpoint this run's weights started from (load_weights), if any.
        self.init_from = init_from
        self.logger: RunLogger | None = None

    # ------------------------------------------------------------- resume

    def resume(self, path: str | Path) -> int:
        """Continue the run ``path`` was saved from, exactly where it stopped.

        Refuses a checkpoint saved under another config (``max_steps`` alone
        changes the schedule) or at another realised precision, and an fp16
        run whose checkpoint lacks the GradScaler's state.
        """
        payload = load_checkpoint(path, map_location=self.spec.device, restore_rng=False)
        saved = flatten(payload.get("config") or {})
        current = flatten(self.cfg.to_dict())
        differing = sorted(k for k in saved.keys() | current.keys() if _canonical(saved.get(k)) != _canonical(current.get(k)))
        if differing:
            shown = ", ".join(f"{k} ({saved.get(k)!r} there, {current.get(k)!r} here)" for k in differing[:5])
            raise ValueError(f"{path} was saved under another config: {shown}")
        saved_precision = payload.get("precision")
        if saved_precision is not None and saved_precision != self.spec.precision:
            raise ValueError(f"{path} was trained in {saved_precision}; this run would continue in "
                             f"{self.spec.precision}")
        saved_init = payload.get("init_from")
        if self.init_from is not None and saved_init != self.init_from:
            raise ValueError(f"{path} started from {saved_init}, not {self.init_from}")
        restore_checkpoint(payload, model=self.model, optimizer=self.optimizer, sampler=self.sampler,
                           scaler=self.scaler, source=path)
        self.state.step = int(payload.get("step", 0))
        self.state.tokens_seen = self.state.step * self.cfg.tokens_per_step
        self.state.best_val_loss = float(payload.get("metrics", {}).get("best_val_loss", math.inf))
        self.init_from = saved_init
        print(f"resumed from {path} at step {self.state.step}")
        return self.state.step

    # --------------------------------------------------------- validation

    @torch.no_grad()
    def validate(self) -> dict[str, float] | None:
        """Each validation set's mean loss over its fixed windows, by name."""
        if not self.val_sets:
            return None
        self.model.eval()
        losses = {}
        for name, batches in self.val_sets.items():
            total = 0.0
            for x, y in batches:
                x, y = x.to(self.spec.device), y.to(self.spec.device)
                with autocast_ctx(self.spec):
                    _, loss = self.model(x, targets=y)
                total += float(loss)
            losses[name] = total / len(batches)
        self.model.train()
        return losses

    # ---------------------------------------------------------- one step

    def _train_step(self) -> tuple[float, float]:
        cfg = self.cfg.training
        self.optimizer.zero_grad(set_to_none=True)

        # Summed on the device: a float() per micro-step would make the CPU
        # wait for the GPU 16 times a flagship step instead of once.
        total_loss = torch.zeros((), device=self.spec.device)
        for _ in range(cfg.gradient_accumulation):
            x, y = self.sampler.batch(self.spec.device)
            with autocast_ctx(self.spec):
                _, loss = self.model(x, targets=y)
            # Scale by the accumulation count so the gradient matches what a
            # single large batch would have produced.
            self.scaler.scale(loss / cfg.gradient_accumulation).backward()
            total_loss += loss.detach()

        self.scaler.unscale_(self.optimizer)
        # clip_grad_norm_ returns the norm *before* clipping, which is the
        # quantity worth logging: it shows spikes that clipping then hides.
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(total_loss) / cfg.gradient_accumulation, grad_norm

    # ------------------------------------------------------------- train

    def train(self, *, stop_at_step: int | None = None, stop_by: datetime | None = None) -> dict:
        """Train to ``max_steps``, or stop cleanly before it.

        After the step that reaches ``stop_at_step``, or the first step to end
        past ``stop_by`` (timezone-aware), the run saves ``last.pt``, logs a
        ``paused`` record, appends this session's cost row and returns
        ``{"paused": True, ...}``.  ``max_steps`` and the schedule are not
        touched, so ``resume`` from ``last.pt`` continues the same run.
        """
        cfg = self.cfg.training
        start_step, start_tokens = self.state.step, self.state.tokens_seen
        if stop_at_step is not None and stop_at_step <= start_step:
            raise ValueError(f"stop_at_step {stop_at_step} is not after this session's first step ({start_step})")
        if stop_by is not None and stop_by.tzinfo is None:
            raise ValueError("stop_by must be timezone-aware")
        device_info = describe(self.spec)
        self.logger = RunLogger(
            self.run_dir, self.cfg.to_dict(), device_info,
            wandb_project=self._wandb_project, resume=start_step > 0,
            data=self.data_info, init_from=self.init_from, start_step=start_step,
        )

        params = self.model.num_parameters()
        flops_per_token = self.model.flops_per_token()
        print(
            f"run '{self.cfg.name}' | {params/1e6:.2f}M params | {device_info['accelerator']} "
            f"| {self.spec.precision} | {self.cfg.tokens_per_step:,} tokens/step "
            f"| budget {self.cfg.token_budget/1e6:.0f}M tokens"
        )

        self.model.train()
        started = time.perf_counter()

        # A validation point before any update, so every curve has a baseline
        # at the untrained model rather than starting mid-descent.
        if self.state.step == 0:
            initial = self.validate()
            if initial is not None:
                first = next(iter(initial.values()))
                self._log({"step": 0, "val_loss": first, "val_ppl": _ppl(first),
                           **self._named(initial), "phase": "init"})
        else:
            self._log({"step": self.state.step, "phase": "resumed"})

        for step in range(self.state.step, cfg.max_steps):
            apply_schedule(
                self.optimizer, self.peak_lrs, step, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio
            )
            lr = lr_at_step(step, cfg.learning_rate, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
            self.meter.start()
            loss, grad_norm = self._train_step()
            self._sync()
            self.meter.stop(self.cfg.tokens_per_step)

            self.state.step = step + 1
            self.state.tokens_seen += self.cfg.tokens_per_step

            if not math.isfinite(loss):
                self._log({"step": self.state.step, "train_loss": loss, "phase": "diverged"})
                raise RuntimeError(
                    f"loss became {loss} at step {self.state.step}; stopping rather than "
                    "burning GPU hours on a diverged run"
                )

            if self.state.step % cfg.log_interval == 0:
                self._log_train(loss, grad_norm, lr, flops_per_token)

            if cfg.validation_interval and self.state.step % cfg.validation_interval == 0:
                self._run_validation(loss)

            if cfg.checkpoint_interval and self.state.step % cfg.checkpoint_interval == 0:
                self.save(f"step_{self.state.step}.pt")
                rotate_checkpoints(self.run_dir, cfg.keep_last_n_checkpoints)

            if self.state.step < cfg.max_steps:
                reason = _pause_reason(self.state.step, stop_at_step, stop_by)
                if reason:
                    return self._pause(reason, started, start_step, start_tokens)

        # Skip the closing validation if the loop just ran one at this step.
        just_validated = (
            cfg.validation_interval and self.state.step % cfg.validation_interval == 0
            and self._last_validation is not None and self._last_validation[0] == self.state.step
        )
        final_losses = self._last_validation[1] if just_validated else self._run_validation(None, force=True)
        final_val = next(iter(final_losses.values())) if final_losses else None
        elapsed = time.perf_counter() - started
        self.save("last.pt")

        summary = {
            "steps": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "final_val_loss": final_val,
            "best_val_loss": self.state.best_val_loss,
            "runtime_seconds": round(elapsed, 1),
            "runtime": format_duration(elapsed),
            "tokens_per_second": round(self.meter.average_tokens_per_second, 1),
            "mfu": self.meter.mfu(flops_per_token, self.spec.name),
            "params": params,
            "realised_mixture": self.sampler.realised_mixture,
            "precision": self.spec.precision,
        }
        if final_losses and len(final_losses) > 1:
            summary["final_val_losses"] = final_losses
        self._record_cost(elapsed, final_val, self.state.tokens_seen - start_tokens,
                          f"resumed at step {start_step}" if start_step else "")
        if self.logger:
            self.logger.finalise(summary)
            self.logger.close()

        print(
            f"done in {summary['runtime']} | {summary['tokens_per_second']:,.0f} tok/s "
            f"| final val loss {final_val if final_val is None else round(final_val, 4)}"
        )
        return summary

    def _pause(self, reason: str, started: float, start_step: int, start_tokens: int) -> dict:
        elapsed = time.perf_counter() - started
        self.save("last.pt")
        self._log({"step": self.state.step, "phase": "paused", "reason": reason})
        stats = {
            "paused": True,
            "steps": self.state.step,
            "reason": reason,
            "tokens_seen": self.state.tokens_seen,
            "runtime_seconds": round(elapsed, 1),
            "tokens_per_second": round(self.meter.average_tokens_per_second, 1),
        }
        note = f"paused at step {self.state.step}"
        self._record_cost(elapsed, None, self.state.tokens_seen - start_tokens,
                          f"resumed at step {start_step}, {note}" if start_step else note)
        if self.logger:
            self.logger.pause(self.state.step, reason, stats)
            self.logger.close()
        print(f"paused at step {self.state.step}/{self.cfg.training.max_steps} ({reason}); "
              f"resume from {self.run_dir / 'last.pt'}")
        return stats

    # ------------------------------------------------------------ helpers

    def _sync(self) -> None:
        """Make the timer measure compute, not queueing.

        CUDA and MPS launches are asynchronous, so without this the first
        timings measure how fast Python can enqueue work.
        """
        if self.spec.type == "cuda":
            torch.cuda.synchronize()
        elif self.spec.type == "mps":
            torch.mps.synchronize()

    def _log(self, record: dict) -> None:
        self.state.history.append(record)
        if self.logger:
            self.logger.log(record)

    def _log_train(self, loss: float, grad_norm: float, lr: float, flops_per_token: float) -> None:
        mfu = self.meter.mfu(flops_per_token, self.spec.name)
        record = {
            "step": self.state.step,
            "train_loss": round(loss, 5),
            "train_ppl": _ppl(loss),
            "grad_norm": round(grad_norm, 4),
            "lr": lr,
            "tokens_seen": self.state.tokens_seen,
            "tokens_per_second": round(self.meter.tokens_per_second, 1),
            "phase": "train",
        }
        muon_lrs = [g["lr"] for g in self.optimizer.param_groups if g.get("use_muon")]
        if muon_lrs:
            # "lr" above is AdamW's, which the Muon arm still uses for the
            # embedding and the norms.
            record["muon_lr"] = muon_lrs[0]
        if mfu is not None:
            record["mfu"] = round(mfu, 4)
        self._log(record)
        print(
            f"step {self.state.step:6d}/{self.cfg.training.max_steps} "
            f"loss {loss:7.4f}  gnorm {grad_norm:6.3f}  lr {lr:.2e}  "
            f"{self.meter.tokens_per_second:8,.0f} tok/s"
            + (f"  mfu {mfu*100:4.1f}%" if mfu is not None else "")
        )

    def _named(self, losses: dict[str, float]) -> dict:
        """``{"val_losses": {...}}`` when there is more than one set, else nothing."""
        if len(losses) < 2:
            return {}
        return {"val_losses": {name: round(loss, 5) for name, loss in losses.items()}}

    def _run_validation(self, train_loss: float | None = None, force: bool = False) -> dict[str, float] | None:
        losses = self.validate()
        if losses is None:
            return None
        val = next(iter(losses.values()))
        improved = val < self.state.best_val_loss
        if improved:
            self.state.best_val_loss = val
            self.save("best.pt")
        self._log({
            "step": self.state.step,
            "val_loss": round(val, 5),
            "val_ppl": _ppl(val),
            "best_val_loss": round(self.state.best_val_loss, 5),
            **self._named(losses),
            "phase": "validation",
        })
        self._last_validation = (self.state.step, losses)
        others = "".join(f"  {name} {loss:.4f}" for name, loss in list(losses.items())[1:])
        print(f"  val loss {val:.4f}  ppl {_ppl(val):8.2f}{others}" + ("  *best" if improved else ""))
        return losses

    def save(self, name: str, extra: dict | None = None) -> Path:
        return save_checkpoint(
            self.run_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            step=self.state.step,
            config=self.cfg.to_dict(),
            sampler_state=self.sampler.state_dict(),
            metrics={
                "best_val_loss": self.state.best_val_loss,
                "tokens_seen": self.state.tokens_seen,
            },
            extra=extra,
            init_from=self.init_from,
            scaler_state=self.scaler.state_dict(),
            precision=self.spec.precision,
        )

    def _record_cost(self, elapsed: float, final_val: float | None, tokens: int, note: str = "") -> None:
        """This session's row: its own runtime and tokens, not the run's so far."""
        try:
            cost_utils.append_cost_row(
                self.costs_path,
                experiment=self.experiment,
                run_id=self.run_dir.name,
                venue=self.venue,
                device=str(self.spec.device),
                gpu_type=self.spec.name,
                runtime_seconds=elapsed,
                tokens_processed=tokens,
                model_params=self.model.num_parameters(),
                tokens_per_second=self.meter.average_tokens_per_second,
                final_val_loss=final_val,
                notes=f"{self.cfg.notes} [{note}]" if note else self.cfg.notes,
            )
        except KeyError as exc:
            # An unknown venue must not destroy a finished run.
            print(f"cost row not written: {exc}")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _pause_reason(step: int, stop_at_step: int | None, stop_by: datetime | None) -> str | None:
    if stop_at_step is not None and step >= stop_at_step:
        return f"stop_at_step {stop_at_step}"
    if stop_by is not None and datetime.now(timezone.utc) >= stop_by:
        return f"stop_by {stop_by.isoformat(timespec='seconds')}"
    return None


def _ppl(loss: float) -> float:
    return round(math.exp(min(loss, 20)), 4)
