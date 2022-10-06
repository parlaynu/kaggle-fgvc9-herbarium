#!/usr/bin/env python3
import sys, os
import re
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", help="location of original images")
    parser.add_argument("out_dir", help="location of resized images")
    parser.add_argument("size_spec", help="{width}x{height} - size to resize (omit one to preserve aspect ratio)", type=str)
    parser.add_argument("crop_spec", help="{width}x{height} - size to centre crop (omit one to preserve aspect ration)", type=str, nargs='?', default=None)
    
    args = parser.parse_args()
    
    srcdir = os.path.expanduser(args.src_dir)
    if not os.path.exists(srcdir) or not os.path.isdir(srcdir):
        print("Error: source directory does not exist")
        sys.exit(1)
        
    outdir = os.path.expanduser(args.out_dir)
    if os.path.exists(outdir):
        print("Error: output directory already exists; won't overwrite")
        sys.exit(1)
    
    size_re = re.compile(r"^(\d*)x(\d*)$")
    
    # process the size spec
    m = size_re.match(args.size_spec)
    if m is None:
        print(f"Error: can't process the size specification: {args.size_spec}")
        sys.exit(1)
    
    r_width = None if len(m.group(1)) == 0 else int(m.group(1))
    r_height = None if len(m.group(2)) == 0 else int(m.group(2))
    
    if r_width is None and r_height is None:
        print("Error: must specify at least one dimension for resize")
        sys.exit(1)

    # process the crop spec
    c_width = c_height = None
    
    if args.crop_spec is not None:
        m = size_re.match(args.crop_spec)
        if m is None:
            print(f"Error: can't process the crop specification: {args.crop_spec}")
            sys.exit(1)
    
        c_width = None if len(m.group(1)) == 0 else int(m.group(1))
        c_height = None if len(m.group(2)) == 0 else int(m.group(2))
    
        if c_width is None and c_height is None:
            print("Error: must specify at least one dimension for crop")
            sys.exit(1)
            
    # create the resizer
    os.makedirs(outdir, mode=0o775, exist_ok=False)
    
    r = Resizer(srcdir, outdir, r_width, r_height, c_width, c_height)
    return r


class Resizer:
    def __init__(self, srcdir, outdir, r_width, r_height, c_width, c_height):
        self.srcdir = srcdir
        self.outdir = outdir
        self.r_width = r_width
        self.r_height = r_height
        self.c_width = c_width
        self.c_height = c_height
        
    def run(self):
        self.scan(self.srcdir, self.outdir)
        
    def scan(self, sroot, oroot):
        # sort entries so can judge progress...
        entries = list(os.listdir(sroot))
        entries.sort()
        
        for entry in entries:
            if entry.startswith("."):
                continue
            
            spath = os.path.join(sroot, entry)
            opath = os.path.join(oroot, entry)
    
            if os.path.isfile(spath):
                if os.path.exists(opath):
                    continue
                if self.process(spath, opath) == False:
                    return False
    
            elif os.path.isdir(spath):
                os.makedirs(opath, mode=0o775, exist_ok=True)
                if self.scan(spath, opath) == False:
                    return False
    
        return True

    def is_jpeg(self, name):
        lname = name.lower()
        if lname.endswith(".jpg") or lname.endswith(".jpeg"):
            return True
        return False

    def process(self, spath, opath):
        print(f"processing {spath}")

        with Image.open(spath) as im:
            swidth, sheight = im.size
            
            # the resize
            rwidth = self.r_width
            if rwidth is None:
                rwidth = int(swidth*self.r_height/sheight)
        
            rheight = self.r_height
            if rheight is None:
                rheight = int(rheight*self.r_width/swidth)
    
            print(f"  resize {swidth}x{sheight} -> {rwidth}x{rheight}")
            out = im.resize((rwidth, rheight), Image.Resampling.BILINEAR)
            
            # the crop
            if self.c_width is not None or self.c_height is not None:
                cwidth = self.c_width
                if cwidth is None:
                    cwidth = int(rwidth*self.c_height/rheight)
        
                cheight = self.c_height
                if cheight is None:
                    cheight = int(rheight*self.c_width/rwidth)
    
                print(f"    crop {rwidth}x{rheight} -> {cwidth}x{cheight}")
                
                left = (rwidth - cwidth)/2
                right = left + cwidth
                upper = (rheight - cheight)/2
                lower = upper + cheight
                
                out = out.crop((left, upper, right, lower))
            
            # the saving
            if self.is_jpeg(opath):
                out.save(opath, quality=95)
            else:
                out.save(opath)
    
        return True


def run():
    r = parse_args()
    r.run()

