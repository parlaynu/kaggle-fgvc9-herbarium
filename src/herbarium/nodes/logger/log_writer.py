import time
from torch.utils.tensorboard import SummaryWriter


class LogWriter:
    def __init__(self, *, log_dir="logs"):
        if log_dir.find("/") == -1:
            now = int(time.time())
            log_dir = f"{log_dir}/{now}"
        self._log_dir = log_dir
        
        self._writer = None
    
    def __getattr__(self, name):
        if self._writer is None:
            self._writer = SummaryWriter(log_dir=self._log_dir)
        
        return getattr(self._writer, name)
    
    def flush(self):
        self._writer.flush()
    
    def close(self):
        self._writer.close()
