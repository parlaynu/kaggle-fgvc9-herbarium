#!/usr/bin/env python

def parse_cmdline():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('state_files', help='state file to load', type=str, nargs='+')
    args = parser.parse_args()
    
    return args.state_files


def scrub_state(state_file):
    import torch
    
    print(f"scrubbing {state_file}")
    
    state = torch.load(state_file, map_location="cpu")
    if 'optimizer' in state:
        del state['optimizer']
    torch.save(state, state_file)


def run():
    state_files = parse_cmdline()
    for state_file in state_files:
        scrub_state(state_file)


if __name__ == "__main__":
    run()



