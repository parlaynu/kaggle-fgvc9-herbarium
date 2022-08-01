from itertools import islice

import torch

import albumentations as A
import albumentations.pytorch

from herbarium.utils import progress

from herbarium.nodes import get_root, iter_fwd
from herbarium.nodes import data
from herbarium.nodes import train
from herbarium.nodes import validate
from herbarium.nodes import evaluate
from herbarium.nodes import logger

import herbarium.model as M


def run():
    train_vdate()


def test_simple():
    batch_size = 16

    tdata = data.HerbariumDataset("~/Projects/datasets/fgvc9-herbarium-2022", "train", batch_size, 
                                            shuffle=False, 
                                            load_images=False)

    print(f"num categories: {tdata.num_categories()}")
    print(f"num images: {len(tdata)}")

    for item in islice(tdata, 2):
        keys = list(item.keys())
        keys.sort()
        
        print("========================================")
        print(len(keys))

        for k in keys:
            if k == "image":
                continue
            print(f"{k} = {item[k]}")

    vdata = data.HerbariumDataset("~/Projects/datasets/fgvc9-herbarium-2022", "val", batch_size, 
                                            shuffle=False, 
                                            load_images=False)

    print(f"num categories: {tdata.num_categories()}")
    print(f"num images: {len(vdata)}")

    for item in islice(vdata, 2):
        keys = list(item.keys())
        keys.sort()
        
        print("========================================")
        print(len(keys))

        for k in keys:
            if k == "image":
                continue
            print(f"{k} = {item[k]}")
    

def train_pipe(batch_size):
    tpipe = dataset = data.HerbariumDataset("~/Projects/datasets/fgvc9-herbarium-2022", "train", batch_size)
    num_categories = tpipe.num_categories()

    #tpipe = data.BatchLimiter(tpipe, 1000, batch_size)
    
    xforms = [
        A.Resize(500, 333),
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ]
    tpipe = data.Transformer(tpipe, xforms)
    tpipe = data.DataLoader(tpipe, batch_size=batch_size, num_workers=6, pin_memory=True, drop_last=True)
    
    model = M.mobilenet_v3_small(dataset.num_categories())
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    tpipe = train.Trainer(tpipe, model, criterion, optimizer)
    
    tpipe = evaluate.F1Score(tpipe, num_categories)
    
    lw = logger.LogWriter()
    tpipe = logger.Logger(tpipe, lw, "Train")
    
    return tpipe, model, lw


def validate_pipe(batch_size, model, lw):
    vpipe = data.HerbariumDataset("~/Projects/datasets/fgvc9-herbarium-2022", "val", batch_size)
    num_categories = vpipe.num_categories()

    #vpipe = data.BatchLimiter(vpipe, 1000, batch_size)
    
    xforms = [
        A.Resize(500, 333),
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ]
    vpipe = data.Transformer(vpipe, xforms)
    vpipe = data.DataLoader(vpipe, batch_size=batch_size, num_workers=6, pin_memory=True, drop_last=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    vpipe = validate.Validator(vpipe, model, criterion)
    
    vpipe = evaluate.F1Score(vpipe, num_categories)
    vpipe = logger.Logger(vpipe, lw, "Vdate")
    
    return vpipe

    
def train_vdate():
    batch_size = 96
    print("building train pipeline...")
    tpipe, model, lw = train_pipe(batch_size)
    
    troot = get_root(tpipe)
    print(troot.num_categories())

    print("train pipeline:")
    for node in iter_fwd(tpipe):
        print(f"- {node.fullname}")

    print("building validate pipeline...")
    vpipe = validate_pipe(batch_size, model, lw)

    print("vdate pipeline:")
    for node in iter_fwd(vpipe):
        print(f"- {node.fullname}")

    
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch:03d}")
        
        tdset = get_root(tpipe)
        tdset.shuffle()

        for idx, item in progress(tpipe, header="Train", end=""):
            pass
        metrics = item['metrics']
        del metrics['lr']
        for k, v in islice(metrics.items(), 3):
            print(f', {k}={v:0.4f}', end='')
        print(flush=True)
        
        with torch.no_grad():
            for idx, item in progress(vpipe, header="Vdate", end=""):
                pass
            metrics = item['metrics']
            for k, v in islice(metrics.items(), 3):
                print(f', {k}={v:0.4f}', end='')
            print(flush=True)
