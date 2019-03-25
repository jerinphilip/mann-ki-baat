import os
import re
import glob
from argparse import ArgumentParser
from collections import defaultdict, Counter

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

glob_str = os.path.join(args.input, '*.src')
files = glob.glob(glob_str)

parallel = defaultdict(set)


required = ['hi', 'ml', 'ta', 'ur', 'te', 'bn']


def extract(value):
    value = value.strip()
    value = value.replace(' ', '')
    value = value.replace('▁', ' ')
    value = value[1:]
    return value


for _file in files:
    basename = os.path.basename(_file)
    tag, ext = os.path.splitext(basename)
    src_lang, tgt_lang = tag.split('-')

    if src_lang in required:
        print(src_lang)
        src_file = open(_file)
        tgt_fname = _file.replace(".src", ".tgt")
        tgt_file = open(tgt_fname)
        for src, tgt in zip(src_file, tgt_file):
            src = extract(src)
            tgt = extract(tgt)
            parallel[tgt].add((src_lang, src))

lengths = []
for tgt in parallel:
    lengths.append(len(parallel[tgt]))
    if len(parallel[tgt]) == 5:
        print(tgt)
        for sample in parallel[tgt]:
            lang, content = sample
            print('\t{}>'.format(lang), content)
        

from pprint import pprint
pprint(Counter(lengths))




