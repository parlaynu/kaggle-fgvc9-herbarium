import os
import random
import numpy as np
import torch
import torch.optim.swa_utils as swa_utils


def save_state(epoch, model, optimizer, name, *, state_dir=None, overwrite=False, verbose=True):

    # make sure the directory exists
    if state_dir is not None and not os.path.exists(state_dir):
        os.makedirs(state_dir)
    
    # create the file name
    state_file = name + ".pt"
    if state_dir is not None:
        state_file = os.path.join(state_dir, state_file)
    
    if verbose:
        print(f"Saving state to {state_file}")
        
    if overwrite == False and os.path.exists(state_file):
        if verbose:
            print("- file exists... skipping")
        return
    
    state = { 'epoch': epoch }
    if model is not None:
        state['model'] = model.state_dict()
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
        
    state['python-random'] = random.getstate()
    state['numpy-random'] = np.random.get_state()
    state['torch-random'] = torch.get_rng_state()
    if torch.cuda.is_available():
        state['torch-cuda-random'] = torch.cuda.get_rng_state()
    
    torch.save(state, state_file)


def load_state(model, optimizer, name, *, state_dir=None, rng_state=True, device=None, verbose=True):
    state_file = name + ".pt"
    if state_dir is not None:
        state_file = os.path.join(state_dir, state_file)
    
    return load_state_file(model, optimizer, state_file, rng_state=rng_state, device=device, verbose=verbose)


def load_state_file(model, optimizer, state_file, *, rng_state=True, device=None, verbose=True):
    if verbose:
        print(f"Loading state from {state_file}")
    
    # get the device
    if device is None:
        device = torch.device('cpu') 
        if model is not None:
            device = next(model.parameters()).device
    
    # wrap the model if it's an swa model... and needs wrapping
    state_file_name = os.path.basename(state_file)
    if '-swa' in state_file_name and not isinstance(model, swa_utils.AveragedModel):
        model = swa_utils.AveragedModel(model)
    
    # load the model, mapping parameters to the device we're on
    state = torch.load(state_file, map_location=device)
    epoch = state['epoch']
    
    if model is not None and 'model' in state:
        model.load_state_dict(state['model'])
    
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])

    if rng_state:
        if (rstate := state.get('python-random', None)) is not None:
            random.setstate(rstate)
        if (rstate := state.get('numpy-random', None)) is not None:
            np.random.set_state(rstate)
        if (rstate := state.get('torch-random', None)) is not None:
            torch.set_rng_state(rstate.type(torch.ByteTensor))
        if torch.cuda.is_available():
            if (rstate := state.get('torch-cuda-random', None)) is not None:
                torch.cuda.set_rng_state(rstate.type(torch.ByteTensor))

    return (epoch, model, optimizer)

