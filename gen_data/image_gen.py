import os
import argparse
import time
from multiprocessing import Pool, Value
import subprocess
import numpy as np
import logging

counter = Value('i', 0)
total = 0

def gen(cmd):
    global counter, total
    subprocess.call(cmd, shell=True)
    with counter.get_lock():
        counter.value += 1
        print("({}/{}) {}".format(counter.value, total, cmd))


def process(args):
    global counter, total
    worker = args.worker
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    else:
        print("ERROR! Output {} already exist!".format(args.output))
        exit(1)
    f_chars = open(args.labellist, mode='r', encoding='utf-8').readline().strip()
    list_chars = []
    for char in f_chars:
        if len(char) > 0:
            list_chars.append(char)
    f_fonts = open(args.fontlist, mode='r', encoding='utf-8').readlines()
    list_fonts = []
    for font in f_fonts:
            list_fonts.append(font.strip())
    print("Working worker: {}".format(worker))
    list_cmd = []
    for char in list_chars:
        for font in list_fonts:
            char_dir = os.path.join(args.output, "label_{}".format(char))
            try:
                os.mkdir(char_dir)
            except:
                pass
            s = 'convert -font "{}" -size 100x100 -gravity center -background black -fill white label:"{}" {}/{}_100.jpg'.format(
                    font, char, char_dir, os.path.basename(font))
            list_cmd.append(s)
            if not "light" in font.lower():
                s = 'convert -font "{}" -size 80x80 -resize 100x100 -gravity center -threshold 50% -background black -fill white label:"{}" {}/{}_80.jpg'.format(
                        font, char, char_dir, os.path.basename(font))
                list_cmd.append(s)

                s = 'convert -font "{}" -size 60x60 -resize 100x100 -gravity center -threshold 50% -background black -fill white label:"{}" {}/{}_60.jpg'.format(
                        font, char, char_dir, os.path.basename(font))
                list_cmd.append(s)

                s = 'convert -font "{}" -size 40x40 -resize 100x100 -gravity center -threshold 50% -background black -fill white label:"{}" {}/{}_40.jpg'.format(
                        font, char, char_dir, os.path.basename(font))
                list_cmd.append(s)

    total = len(list_cmd)
    p = Pool(worker)
    p.map(gen, list_cmd)
        

parser = argparse.ArgumentParser(description="")
parser.add_argument("--fontlist", type=str, help="Text file containing list of font's path", required=True)
parser.add_argument("--labellist", type=str, help="Text file containing sequence of charactor label", required=True)
parser.add_argument("--output", type=str, help="Output folder", required=True)
parser.add_argument("--worker", type=int, default=1)
args = parser.parse_args()

process(args)
