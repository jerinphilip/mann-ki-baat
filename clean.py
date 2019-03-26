import os
import re
import glob
from argparse import ArgumentParser
from collections import defaultdict, Counter
import pf
from pf.dataset import ParallelWriter
from pprint import pprint
from pf.sentencepiece import SentencePieceTokenizer

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

glob_str = os.path.join(args.input, '*.src')
files = glob.glob(glob_str)

parallel = defaultdict(set)
tokenizer = SentencePieceTokenizer()


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
    keys = list(lang_dict.keys())
    translations = []
    for xx in keys:
        for yy in keys:
            translation = Translation(
                src_lang=xx, 
                tgt_lang=yy, 
                src=lang_dict[xx], 
                tgt=lang_dict[yy]
            )
            translations.append(translation)
    return translations


pairs = []
lengths = []
for tgt in parallel:
    entry = {}
    lengths.append(len(parallel[tgt]))
    if len(parallel[tgt]) == 5:
        entry['en'] = tgt
        for sample in parallel[tgt]:
            lang, content = sample
            entry[lang] = content

        pairs.extend(all_pairs(entry))
        # for translation in all_pairs(entry):
        #     pairs.extend(translation)

        #     print(translation)


mapping_fpath = os.path.join(args.output, 'test.mapping')
mapping_file = open(mapping_fpath, 'w+')
writer = ParallelWriter(args.output,  'test', 'src', 'tgt')
for idx, sample in enumerate(pairs):
    src_token = pf.utils.language_token(sample.src_lang)
    tgt_token = pf.utils.language_token(sample.tgt_lang)
    print(idx, src_token, tgt_token, file=mapping_file)

    lang, src_tokens = tokenizer(sample.src)
    src = [tgt_token] + src_tokens
    src = ' '.join(src)

    lang, tgt_tokens = tokenizer(sample.tgt)
    tgt = ' '.join(tgt_tokens)

    writer.write(src, tgt)

mapping_file.close()
