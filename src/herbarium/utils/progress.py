import math
import time

def progress(iterable, *, header=None, hashes=50, end="\n"):
    
    batch_total = len(iterable)
    hash_count = 0
    
    start = time.time()

    if header is not None:
        print(f"{header}: ", end="", flush=True)

    for batch, i in enumerate(iterable):
        yield batch, i
        hash_target = math.floor((batch+1)/batch_total*hashes)
        while hash_count < hash_target:
            hash_count += 1
            print("#", end="", flush=True)
    
    elapsed = (time.time() - start)/60
    
    print(f": b={batch+1:04d}, t={elapsed:0.2f}m", end=end, flush=True)

