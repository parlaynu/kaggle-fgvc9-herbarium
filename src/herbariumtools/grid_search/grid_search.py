import os, sys
import argparse
import random
import yaml
import hashlib
import subprocess


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--run-limit', help='max trials to run', type=int, default=10000)
    parser.add_argument('-x', '--execute', help='the executable to run', type=str, default='train')
    parser.add_argument('-c', '--use-cpu', help='use the CPU even if there is a GPU', action='store_true')
    parser.add_argument('-t', '--time-limit', help='max runtime in minutes (0 = no limit)', type=int, default=0)
    parser.add_argument('-e', '--num-epochs', help='number of epochs (0 = no limit)', type=int, default=0)
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=0)
    parser.add_argument('-w', '--num-workers', help='number of workers to use', type=int, default=0)
    
    parser.add_argument('config_file', help='configuration file to load', type=str, default=None)
    parser.add_argument('vars_file', help='variables for template expansion', type=str, default=None)
    
    args = parser.parse_args()

    cmdline = [args.execute]
    if args.use_cpu:
        cmdline.append('-c')
    if args.time_limit > 0:
        cmdline += ['-t', str(args.time_limit)]
    if args.num_epochs > 0:
        cmdline += ['-e', str(args.num_epochs)]
    if args.batch_size > 0:
        cmdline += ['-b', str(args.batch_size)]
    if args.num_workers > 0:
        cmdline += ['-w', str(args.num_workers)]
    cmdline.append(args.config_file)
    
    return (args.run_limit, args.vars_file, cmdline)


def load_vars(vars_file):
    with open(vars_file, 'r') as vf:
        return yaml.safe_load(vf)
    
    
def run():
    # parse the command line and load the vars
    run_limit, vars_file, cmdline = parse_cmdline()
    allvars = load_vars(vars_file)
    
    # make sure the hash directory exists
    hashdir = "hashes"
    os.makedirs(hashdir, mode=0o777, exist_ok=True)
    
    # find a unique commandline
    for i in range(run_limit):
        options = []
        for vname, voptions in allvars.items():
            voption = random.choice(voptions)
            
            # voption is expected to be a templated value in a yaml file. To pass through
            # correctly, None in python is Null in yaml.
            if voption is None:
                voption = "Null"
            
            options.append(f"{vname}={voption}")
            
            if isinstance(voption, str) and ',' in voption:
                voptions = voption.split(',')
                for idx, v in enumerate(voptions):
                    options.append(f"{vname}_{idx}={v}")
            
            elif isinstance(voption, (list, tuple)):
                for idx, v in enumerate(voption):
                    options.append(f"{vname}_{idx}={v}")
            
            elif isinstance(voption, dict):
                for k, v in voption.items():
                    options.append(f"{vname}_{k}={v}")
        
        fullcmdline = cmdline + options
        fullcmdstr = " ".join(fullcmdline)
        
        # check the hash of the cmdline
        cmdhash = hashlib.sha256(fullcmdstr.encode('utf-8')).hexdigest()
        fullcmdline.append(f"command_hash={cmdhash}")
        
        cmdhashfile = os.path.join(hashdir, cmdhash)
        if os.path.exists(cmdhashfile):
            fullcmdline = None
            continue
        
        with open(cmdhashfile, 'w') as chf:
            print(fullcmdstr, file=chf)
        
        # run the command
        print(fullcmdstr)
        subprocess.run(fullcmdline)
        
        # reload the vars... in case of edits during previous run
        allvars = load_vars(vars_file)
    
    print("no more unique options")


