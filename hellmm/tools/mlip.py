"""MLIP loader — reusable across projects.

Dispatches to the appropriate calculator backend based on model name prefix:

  fairchem (UMA / eSEN)  — prefix: 'uma-', 'esen-', 'allscaip-'
  MACE-MP / MACE-OFF     — prefix: 'mace-mp', 'mace-off'
"""

from __future__ import annotations

import warnings

_DEFAULT_MLIP = "esen-sm-conserving-all-oc25"

_FAIRCHEM_PREFIXES = ("uma-", "esen-", "allscaip-")
_MACE_PREFIXES = ("mace-mp", "mace-off")


def _best_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_mlip(model: str = _DEFAULT_MLIP, device: str | None = None):
    """Load a universal MLIP as an ASE Calculator.

    Dispatches to the appropriate backend based on the model name prefix.
    Device is auto-detected (cuda > mps > cpu) unless explicitly specified.

      fairchem (UMA / eSEN)  — prefix: 'uma-', 'esen-', 'allscaip-'
        e.g. 'uma-s-1p2', 'esen-sm-conserving-all-oc25'
        Requires HuggingFace login + gated-repo acceptance.
        Note: fairchem only supports cuda/cpu (not mps).

      MACE-MP / MACE-OFF     — prefix: 'mace-mp', 'mace-off'
        e.g. 'mace-mp-medium-omat-0', 'mace-mp-medium-0b2'
        Supports cuda, mps, and cpu.
        Full name list: small, medium, large, small-0b2, medium-0b2,
        large-0b2, medium-0b3, medium-mpa-0, small-omat-0, medium-omat-0,
        mace-matpes-pbe-0, mace-matpes-r2scan-0

    Args:
        model: model identifier string. Default: 'mace-mp-medium-omat-0'.
        device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.

    Returns:
        ASE Calculator

    Raises:
        ValueError: if the model name prefix is not recognised
        PermissionError: for fairchem models if HuggingFace access is not granted
        ImportError: if the required backend package is not installed
    """
    if device is None:
        device = _best_device()

    if any(model.startswith(p) for p in _FAIRCHEM_PREFIXES):
        # fairchem does not support mps — fall back to cpu
        if device == "mps":
            device = "cpu"
        return _load_fairchem(model, device)
    if any(model.startswith(p) for p in _MACE_PREFIXES):
        return _load_mace(model, device)
    raise ValueError(
        f"Unrecognised model '{model}'. "
        f"Fairchem prefixes: {_FAIRCHEM_PREFIXES}. "
        f"MACE prefixes: {_MACE_PREFIXES}."
    )


def _load_fairchem(model: str, device: str):
    """Load a fairchem UMA/eSEN model as a FAIRChemCalculator.

    eSEN models (esen-*) use the newer API: FAIRChemCalculator(predict_unit)
    with no task_name — they have a single output head trained on OC25.
    """
    try:
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
    except ImportError as e:
        raise ImportError("fairchem-core is not installed: pip install fairchem-core") from e

    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Some fairchem/e3nn dependencies store slice objects in .pt files
    # (e.g. Wigner-3j constants.pt) — allowlist slice so the load succeeds.
    import torch
    try:
        torch.serialization.add_safe_globals([slice])
    except AttributeError:
        pass  # PyTorch < 2.6 — method doesn't exist, no-op

    try:
        predict_unit = pretrained_mlip.get_predict_unit(model, device=device)
    except Exception as e:
        e_str = str(e).lower()
        if "401" in str(e) or "403" in str(e) or "gated" in e_str or "access" in e_str:
            # eSEN models live under facebook/OC25; UMA models under facebook/UMA
            hf_repo = "facebook/OC25" if model.startswith("esen-") else "facebook/UMA"
            raise PermissionError(
                f"Cannot download fairchem model '{model}': access denied.\n"
                f"1. Accept the license at https://huggingface.co/{hf_repo}\n"
                "2. Run: hf login"
            ) from e
        raise
    return FAIRChemCalculator(predict_unit)


def _load_mace(model: str, device: str):
    """Load a MACE-MP model as an ASE Calculator.

    The model string uses the prefix 'mace-mp-' followed by the mace_mp
    internal name, e.g. 'mace-mp-medium-omat-0' → internal name 'medium-omat-0'.
    Bare 'mace-mp' uses the default (medium) model.
    """
    try:
        from mace.calculators import mace_mp
    except ImportError as e:
        raise ImportError("mace-torch is not installed: pip install mace-torch") from e

    internal = model[len("mace-mp-"):] if model != "mace-mp" else None
    # MPS does not support float64. Try float32 on MPS; if the checkpoint
    # contains float64 tensors (e.g. omat-0) it will still fail — fall back to cpu.
    if device == "mps":
        try:
            return mace_mp(model=internal, device="mps", dispersion=False, default_dtype="float32")
        except Exception:
            warnings.warn(
                f"Model '{model}' could not load on MPS (likely float64 checkpoint). "
                "Falling back to CPU."
            )
            device = "cpu"
    return mace_mp(model=internal, device=device, dispersion=False, default_dtype="float64")
