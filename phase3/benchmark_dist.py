import time
import nltk.metrics.distance as dist

t0 = time.time()
for _ in range(1000):
    s1 = "walmart supercenter"
    s2 = "wal-mart store inc"
    d1 = dist.edit_distance(s1, s2)
    d2 = dist.jaro_similarity(s1, s2)

elapsed = time.time() - t0
print(f"1000 comparisons took {elapsed:.4f}s ({1000/elapsed:.0f} pairs/sec)")
