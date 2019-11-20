import os
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
import langid
import ilmulti
import pandas as pd
from tqdm import tqdm

keys = ['hi', 'ta', 'ur', 'ml', 'bn', 'te', 'or', 'gj', 'pa', 'mr']

segmenter = ilmulti.segment.Segmenter()

def relaxed_canonicalize(langname):
    mapping = {'gj': 'gu', 'mp': 'bn'}
    return mapping.get(langname, langname)

def lang_condition(tgt_lang, content):
    pred_lang, prob = langid.classify(content)
    cpred_lang = relaxed_canonicalize(pred_lang)
    clang = relaxed_canonicalize(tgt_lang)
    if clang != cpred_lang:
        return False
        # print(lang, clang, pred_lang, cpred_lang, prob)
    return True

def sentence_count(filenames):
    groups = defaultdict(list)
    segments = defaultdict(list)
    for fname in tqdm(filenames):
        fn, ext = fname.split('.')
        lang, idx = fn.split('-')
        fpath = os.path.join(args.dir, fname)
        with open(fpath) as fp:
            content = fp.read()
            lines = content.splitlines()
            cleaned = []
            for line in lines:
                nl = line.lstrip().rstrip()
                cleaned.append(nl)
            
            # Count detect
            final = '\n'.join(cleaned)
            if lang_condition(lang, final):
                groups[lang].append(idx)

                # detect segments
                _lang, subsegments = segmenter(final)
                num_segments = len(subsegments)
                segments[lang].append(num_segments)

    def gen_ordered(redn_f, d):
        ordered_groups = OrderedDict()
        for key in keys:
            vals = d.get(key, [])
            ordered_groups[key] = redn_f(vals)
        return ordered_groups

    

    ls = [
        gen_ordered(len, groups),
        gen_ordered(sum, segments)
    ]

    df = pd.DataFrame.from_dict(ls)
    print(df)


def main(args):
    filenames = os.listdir(args.dir)
    sentence_count(filenames)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
