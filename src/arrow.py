import os
import psutil
import timeit

from datasets import load_dataset


# Load the dataset into memory
mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
wiki = load_dataset("wikipedia", "20220301.en", split="train")
mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
print(f"RAM memory used: {(mem_after - mem_before)} MB")

s = """
batch_size = 1000
for batch in wiki.iter(batch_size):
    ...
"""

elapsed_time = timeit.timeit(stmt=s, number=1, globals=globals())
print(f"Time to iterate over the {wiki.dataset_size >> 30} GB dataset: {elapsed_time:.1f} sec, " f"ie. {float(wiki.dataset_size >> 27)/elapsed_time:.1f} Gb/s")
# RAM memory used: 442.0 MB
# Time to iterate over the 18 GB dataset: 77.9 sec, ie. 1.9 Gb/s