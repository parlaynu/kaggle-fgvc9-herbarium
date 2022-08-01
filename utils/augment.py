#!/usr/bin/env python3
import os, os.path
from itertools import islice

import albumentations as A
from PIL import Image
import numpy as np
import herbarium.nodes.data


print("creating dataset...")
pipe = herbarium.nodes.data.HerbariumDataset("~/Projects/datasets/fgvc9-herbarium-2022", "train", 1, image_dir="train_images_half")

print("creating transforms...")
transform = A.Compose([
    A.Flip(p=0.5),
    A.Resize(height=500, width=333),
    A.CenterCrop(height=350, width=266),
    A.RandomCrop(height=224, width=224),
    A.GaussNoise(p=1.0),
    A.CoarseDropout(p=1.0, max_holes=32, min_holes=16, max_width=24, max_height=24, min_width=8, min_height=8, fill_value=175)
])
pipe = herbarium.nodes.data.AblumentationsTransformer(pipe, transform)


outdir = "outputs/augment"
os.makedirs(outdir, exist_ok=True)

for item in islice(pipe, 10):
    print(f"processing {item['image_id']}")
    
    path = os.path.join(outdir, f"{item['image_id']}.jpg")
    pimg = Image.fromarray(item['image'])
    pimg.save(path)
    
# # load the image
# print("- loading image")
# image_path = os.path.expanduser("~/Projects/datasets/fgvc9-herbarium-2022/train_images_half/000/00/00000__001.jpg")
# image = Image.open(image_path)
# image = np.array(image)
#
# # apply the transform
# print("- applying first transform")
# transformed = transform(image=image)
# timage = transformed['image']
#
# simage = Image.fromarray(timage)
# simage.save("tformed.jpg")

