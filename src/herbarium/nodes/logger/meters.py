
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, val, count=1):
        self._sum += val * count
        self._count += count
    
    def value(self):
        return self._sum / self._count

