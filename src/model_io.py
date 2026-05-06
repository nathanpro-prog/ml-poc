"""Helpers for loading and saving serialized models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Load (provided in the contract + extended)
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> Any:
    """Load a serialized model from disk.

    Supported formats are `.joblib`, `.pkl`, and `.pickle`.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    suffix = model_path.suffix.lower()

    if suffix == ".joblib":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "Loading `.joblib` files requires the `joblib` package. "
                "Add it to requirements.txt if needed."
            ) from exc
        return joblib.load(model_path)

    if suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as file_handle:
            return pickle.load(file_handle)

    raise ValueError(
        f"Unsupported model format for {model_path}. Use .joblib, .pkl, or .pickle."
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    model_path: Path | str,
    *,
    overwrite: bool = True,
) -> Path:
    """Serialize a fitted model to disk.

    Supported formats are `.joblib`, `.pkl`, and `.pickle` — inferred
    from the file extension.

    Parameters
    ----------
    model:
        Any fitted scikit-learn-compatible estimator.
    model_path:
        Destination path, including the file extension.
    overwrite:
        If ``False`` and the file already exists, raises ``FileExistsError``.
        Defaults to ``True``.

    Returns
    -------
    Path
        The resolved path where the model was saved.

    Examples
    --------
    >>> save_model(clf, "models/engine1/random_forest.pkl")
    PosixPath('models/engine1/random_forest.pkl')
    """
    model_path = Path(model_path)

    if not overwrite and model_path.exists():
        raise FileExistsError(
            f"Model file already exists: {model_path}. "
            "Pass overwrite=True to replace it."
        )

    # Create parent directories if needed
    model_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = model_path.suffix.lower()

    if suffix == ".joblib":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "Saving `.joblib` files requires the `joblib` package. "
                "Add it to requirements.txt if needed."
            ) from exc
        joblib.dump(model, model_path)

    elif suffix in {".pkl", ".pickle"}:
        with model_path.open("wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        raise ValueError(
            f"Unsupported model format for {model_path}. "
            "Use .joblib, .pkl, or .pickle."
        )

    return model_path.resolve()


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def save_engine_models(
    models: dict[str, Any],
    engine_dir: Path | str,
    fmt: str = "pkl",
    overwrite: bool = True,
) -> dict[str, Path]:
    """Save a dictionary of models into an engine directory.

    Parameters
    ----------
    models:
        Mapping of ``{model_name: fitted_model}``.
        Model names are sanitized (spaces -> underscores, lower-cased)
        to build file names.
    engine_dir:
        Directory where the models are saved, e.g.
        ``"models/engine1"`` or ``"models/engine2"``.
    fmt:
        File format, one of ``"pkl"``, ``"pickle"``, ``"joblib"``.
    overwrite:
        Passed through to ``save_model``.

    Returns
    -------
    dict[str, Path]
        Mapping of ``{model_name: saved_path}``.

    Examples
    --------
    >>> paths = save_engine_models(
    ...     {"logistic_regression": lr, "random_forest": rf},
    ...     engine_dir="models/engine1",
    ... )
    """
    engine_dir = Path(engine_dir)
    saved: dict[str, Path] = {}

    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        dest      = engine_dir / f"{safe_name}.{fmt.lstrip('.')}"
        saved[name] = save_model(model, dest, overwrite=overwrite)

    return saved


def load_engine_models(
    engine_dir: Path | str,
    fmt: str = "pkl",
) -> dict[str, Any]:
    """Load all models from an engine directory.

    Parameters
    ----------
    engine_dir:
        Directory that contains the serialized models.
    fmt:
        File extension to glob for (without the leading dot).

    Returns
    -------
    dict[str, Any]
        Mapping of ``{stem_name: model}`` for every matching file found.

    Examples
    --------
    >>> models = load_engine_models("models/engine1")
    >>> models.keys()
    dict_keys(['logistic_regression', 'random_forest', 'xgboost'])
    """
    engine_dir = Path(engine_dir)

    if not engine_dir.is_dir():
        raise FileNotFoundError(
            f"Engine directory does not exist: {engine_dir}"
        )

    model_files = list(engine_dir.glob(f"*.{fmt.lstrip('.')}"))

    if not model_files:
        raise FileNotFoundError(
            f"No .{fmt} files found in {engine_dir}."
        )

    return {p.stem: load_model(p) for p in sorted(model_files)}