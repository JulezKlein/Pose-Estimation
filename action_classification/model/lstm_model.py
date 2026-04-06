from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn


class ActionClassifier(nn.Module):
    """LSTM-based binary/multi-class action classifier.

    Parameters
    ----------
    input_size:
        Number of input features per time-step (e.g. 17 keypoints × 3 = 51).
    hidden_size:
        Number of LSTM hidden units.
    num_layers:
        Number of stacked LSTM layers.
    output_size:
        Number of output classes (1 for binary with BCEWithLogitsLoss,
        ≥2 for multi-class with CrossEntropyLoss).
    dropout:
        Dropout probability applied between LSTM layers (ignored when
        num_layers == 1).
    """

    # Hyperparameter keys persisted in every checkpoint.
    _HPARAM_KEYS = ("input_size", "hidden_size", "num_layers", "output_size", "dropout")

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        Logits of shape ``(batch, output_size)``.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def save(self, path: Union[str, Path]) -> None:
        """Save model weights **and** hyperparameters to *path* (``.pt``).

        The checkpoint is a plain dict so it can be loaded without
        re-instantiating the class first (see :meth:`from_ckpt`).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hparams": {k: getattr(self, k) for k in self._HPARAM_KEYS},
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def from_ckpt(
        cls,
        path: Union[str, Path],
        map_location: Union[str, torch.device, None] = None,
    ) -> "ActionClassifier":
        """Load a model from a checkpoint saved by :meth:`save`.

        Parameters
        ----------
        path:
            Path to the ``.pt`` checkpoint file.
        map_location:
            Passed to :func:`torch.load` (e.g. ``"cpu"`` when loading a
            GPU checkpoint on a CPU-only machine).

        Returns
        -------
        A fully initialised :class:`ActionClassifier` with weights restored.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    @classmethod
    def from_config(cls, cfg) -> "ActionClassifier":
        """Instantiate from an OmegaConf ``DictConfig`` (or plain dict).

        Accepts either the full config object (with a ``model`` sub-key) or
        just the model block directly::

            cfg = OmegaConf.load("config.yml")
            model = ActionClassifier.from_config(cfg)        # full config
            model = ActionClassifier.from_config(cfg.model)  # model block only
        """
        from omegaconf import OmegaConf

        model_cfg = cfg.model if hasattr(cfg, "model") else cfg
        hparams = OmegaConf.to_container(model_cfg, resolve=True)
        return cls(**{k: hparams[k] for k in cls._HPARAM_KEYS})

    # ---------------------------------------------------------------------- #
    # Export
    # ---------------------------------------------------------------------- #

    def export_onnx(
        self,
        path: Union[str, Path],
        seq_len: int,
        batch_size: int = 1,
        opset_version: int = 17,
    ) -> None:
        """Export the model to ONNX format.

        Parameters
        ----------
        path:
            Destination ``.onnx`` file.
        seq_len:
            Sequence length used for the dummy input shape.
        batch_size:
            Batch size for the dummy input (default 1).
        opset_version:
            ONNX opset version to target.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(batch_size, seq_len, self.input_size)
        torch.onnx.export(
            self,
            dummy,
            str(path),
            input_names=["keypoints"],
            output_names=["logits"],
            dynamic_axes={
                "keypoints": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch"},
            },
            opset_version=opset_version,
        )

    def export_coreml(
        self,
        path: Union[str, Path],
        seq_len: int,
        minimum_deployment_target: str = "iOS16",
    ) -> None:
        """Export the model to Core ML format (``.mlpackage``).

        Requires ``coremltools>=7.0`` (macOS / Linux).

        Parameters
        ----------
        path:
            Destination ``.mlpackage`` bundle path.
        seq_len:
            Sequence length used for the trace dummy input.
        minimum_deployment_target:
            Core ML deployment target passed to ``ct.convert``
            (e.g. ``"iOS16"``, ``"macOS13"``).
        """
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                "coremltools is required for Core ML export. "
                "Install it with: pip install coremltools"
            ) from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()
        dummy = torch.zeros(1, seq_len, self.input_size)
        traced = torch.jit.trace(self, dummy)

        target = getattr(ct.target, minimum_deployment_target)
        ml_model = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="keypoints",
                    shape=(1, seq_len, self.input_size),
                    dtype=float,
                )
            ],
            outputs=[ct.TensorType(name="logits")],
            minimum_deployment_target=target,
        )
        ml_model.save(str(path))
