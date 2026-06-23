import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """
    Substituto do print() que escreve simultaneamente no console e em um arquivo .txt.

    Uso:
        logger = TrainingLogger("generated/logs")
        logger.log("Epoch 1 accuracy: 85.00%")
        logger.close()

        # Ou usando context manager:
        with TrainingLogger("generated/logs") as logger:
            logger.log("Epoch 1 accuracy: 85.00%")
    """

    def __init__(self, log_dir: str = "generated/logs", log_name: str = "training"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = Path(log_dir) / f"{log_name}_{timestamp}.txt"
        self._file = open(self.log_path, "w", encoding="utf-8", buffering=1)  # buffering=1 → flush por linha
        self.log(f"Log iniciado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Arquivo: {self.log_path}\n")

    def log(self, *args, **kwargs):
        """Mesma assinatura do print()."""
        # Imprime no console normalmente
        print(*args, **kwargs)
        # Escreve no arquivo
        print(*args, **kwargs, file=self._file)

    def close(self):
        if not self._file.closed:
            self.log(f"\nLog encerrado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._file.close()

    # Context manager
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()