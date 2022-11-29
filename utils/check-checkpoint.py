#!/usr/bin/env python

def parse_cmdline():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('state_files', help='state file to load', type=str, nargs='+')
    args = parser.parse_args()
    
    return args.state_files


def check_state(state_file):
    import torch
    
    print(f"checking {state_file}")
    
    state = torch.load(state_file, map_location="cpu")
    model = state['model']
    
    for k, v in model.items():
        print(f"{k}: {type(v)}")


def run():
    state_files = parse_cmdline()
    for state_file in state_files:
        check_state(state_file)


if __name__ == "__main__":
    run()



