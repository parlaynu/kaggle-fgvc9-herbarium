#!/usr/bin/env python3
from herbarium.nodes.data import HerbariumDataset


dsroot = "~/Projects/datasets/fgvc9-herbarium-2022"
split = "train"
batch_size = 32

ds = HerbariumDataset(dsroot, split, batch_size)

categories = ds.categories()

print("label,name")
keys = list(categories.keys())
keys.sort()

for k in keys:
    print(f"{k},{categories[k]['label']}")
