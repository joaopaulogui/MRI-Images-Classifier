import sys
from datetime import datetime
from pathlib import Path


class Logger:

    def __init__(self, log_dir: str = "generated/logs", log_name: str = "training"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = Path(log_dir) / f"{log_name}_{timestamp}.txt"
        self._file = open(self.log_path, "w", encoding="utf-8", buffering=1)  
        self.log(f"Log iniciado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Arquivo: {self.log_path}\n")

    def log(self, *args, **kwargs):

        print(*args, **kwargs)

        print(*args, **kwargs, file=self._file)

    def close(self):
        if not self._file.closed:
            self.log(f"\nLog encerrado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()