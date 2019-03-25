import os
import re
import glob
from argparse import ArgumentParser
from collections import defaultdict, Counter
from pprint import pprint

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

from collections import namedtuple

Translation = namedtuple('Translation', 'src src_lang tgt tgt_lang')

def all_pairs(lang_dict):
    # n^2
    keys = list(lang_dicts.keys())
    for xx in keys:
        for yy in keys:
            translation = Translation(src=xx, tgt=yy, src_lang=lang_dict[xx], tgt_lang=lang_dict[yy])

    return Translation

    

lengths = []
for tgt in parallel:
    lang_dicts = {}
    lengths.append(len(parallel[tgt]))
    if len(parallel[tgt]) == 5:
        # print(tgt)
        lang_dicts['en'] = tgt
        for sample in parallel[tgt]:
            lang, content = sample
            # print('\t{}>'.format(lang), content)
            lang_dicts[lang] = content
        pprint(lang_dicts)   

pprint(Counter(lengths))


n2_pairs = all_pairs(lang_dicts)
