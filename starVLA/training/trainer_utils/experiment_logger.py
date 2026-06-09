import numbers
import os
from pathlib import Path

import wandb
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Minimal experiment logger with pluggable backends."""

    SUPPORTED_BACKENDS = {"wandb", "tensorboard", "none"}

    def __init__(self, config=None, logger=None, cfg=None):
        self.config = config or cfg
        if self.config is None:
            raise ValueError("ExperimentLogger requires a config object.")
        self.logger = logger
        self.backend = self._resolve_backend()
        self.output_dir = Path(self.config.output_dir)
        self.tb_writer = None

    def _resolve_backend(self) -> str:
        backend = None
        if hasattr(self.config, "logging_backend"):
            backend = self.config.logging_backend
        elif hasattr(self.config, "trainer") and hasattr(self.config.trainer, "logging_backend"):
            backend = self.config.trainer.logging_backend

        backend = str(backend or "wandb").lower()
        if backend not in self.SUPPORTED_BACKENDS:
            supported = ", ".join(sorted(self.SUPPORTED_BACKENDS))
            raise ValueError(f"Unsupported logging backend `{backend}`. Expected one of: {supported}.")
        return backend

    def init(self):
        if self.backend == "none":
            if self.logger is not None:
                self.logger.info("Experiment logging disabled.")
            return

        if self.backend == "wandb":
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )
            if self.logger is not None:
                self.logger.info("Initialized experiment logging with wandb.")
            return

        tb_dir = self.output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
        if self.logger is not None:
            self.logger.info(f"Initialized experiment logging with TensorBoard at {tb_dir}.")

    def log_metrics(self, metrics, step: int):
        if self.backend == "none":
            return

        if self.backend == "wandb":
            wandb.log(metrics, step=step)
            return

        for key, value in metrics.items():
            if isinstance(value, numbers.Number):
                self.tb_writer.add_scalar(key, value, step)
        self.tb_writer.flush()

    def close(self):
        if self.backend == "wandb":
            wandb.finish()
        elif self.backend == "tensorboard" and self.tb_writer is not None:
            self.tb_writer.close()
