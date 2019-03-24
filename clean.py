import os
import re
import glob
from argparse import ArgumentParser
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

glob_str = os.path.join(args.input, '*.aligned')
files = glob.glob(glob_str)

class Alignment:
    def __init__(self, fpath):
        self.fpath = fpath
        self._set_meta(fpath)

    def _set_meta(self, fpath):
        fname = os.path.basename(fpath)
        tokens = re.split('[_\-.]', fname)
        iidx, _dump, src, sidx, _ext, *rest = tokens
        _dump, tgt, tidx, _ext, *rest = rest
        self.uid = '{}.{}-{}.{}'.format(iidx, src, tgt, sidx)
        self.iidx = iidx
        self.src = src
        self.tgt = tgt
        self.idx = sidx

    def corpus_idx(self):
        key = '{}.{}'.format(self.idx, self.iidx)
        return key

    def __repr__(self):
        return self.uid

    def parse(self):
        samples = []
        with open(self.fpath) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    entries = line.split('\t')
                    if len(entries) == 2:
                        samples.append(entries)
        return samples

# group files
weak_parallel = defaultdict(list)
for _file in files:
    a = Alignment(_file)
    weak_parallel[a.corpus_idx()].append(a)

for key in weak_parallel:
    intersection = defaultdict(list)
    for align in weak_parallel[key]:
        samples = align.parse()
        for src, tgt in samples:
            intersection[tgt].append((a.src, src))

    print("Key:", key)
    for key in intersection:
        if len(intersection[key]) > 5:
            print(key, len(intersection[key]))
    
