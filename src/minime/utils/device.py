"""Device and precision selection.

Three backends matter for this project and they do not agree on anything:

* CUDA Ampere+ (rented 4090/A10): bf16 autocast, no loss scaler needed.
* CUDA Turing (Kaggle/Colab T4): bf16 is emulated and slow, so fp16 + GradScaler.
* Apple MPS (the development laptop): autocast is unreliable, so fp32.

Precision is therefore auto-detected but always recorded in the run manifest,
because it is a confound if it ever silently differs between two arms.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    precision: str            # "fp32" | "fp16" | "bf16"
    name: str                 # human-readable accelerator name
    use_scaler: bool

    @property
    def type(self) -> str:
        return self.device.type


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _accelerator_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def native_bf16(device: torch.device) -> bool:
    """Whether this GPU computes in bf16 in hardware: Ampere (sm_80) and newer.

    Not ``torch.cuda.is_bf16_supported()``, which by default also counts
    emulation and so says yes on a T4 (sm_75).  The E4 AdamW sweep ran that
    way on Kaggle: emulated bf16 at ~16k tok/s, 2% MFU.
    """
    return torch.cuda.get_device_capability(device)[0] >= 8


def resolve(requested_device: str = "auto", requested_precision: str = "auto") -> DeviceSpec:
    device = pick_device(requested_device)
    precision = requested_precision

    if precision == "auto":
        if device.type == "cuda":
            precision = "bf16" if native_bf16(device) else "fp16"
        else:
            precision = "fp32"

    if precision != "fp32" and device.type != "cuda":
        # Autocast on MPS/CPU is either unsupported or slower than fp32 here.
        precision = "fp32"

    return DeviceSpec(
        device=device,
        precision=precision,
        name=_accelerator_name(device),
        use_scaler=(precision == "fp16"),
    )


def autocast_ctx(spec: DeviceSpec) -> Any:
    if spec.precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if spec.precision == "bf16" else torch.float16
    return torch.autocast(device_type=spec.device.type, dtype=dtype)


def make_scaler(spec: DeviceSpec):
    """GradScaler only does anything for fp16; it is a no-op elsewhere."""
    return torch.amp.GradScaler(spec.device.type, enabled=spec.use_scaler)


def describe(spec: DeviceSpec) -> dict[str, Any]:
    info = {
        "device": str(spec.device),
        "accelerator": spec.name,
        "precision": spec.precision,
        "torch": torch.__version__,
    }
    if spec.device.type == "cuda":
        props = torch.cuda.get_device_properties(spec.device)
        info["cuda_capability"] = f"{props.major}.{props.minor}"
        info["vram_gb"] = round(props.total_memory / 1024**3, 1)
    return info
