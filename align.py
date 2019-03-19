from __future__ import unicode_literals
from argparse import ArgumentParser
import os
from polyglot_tokenizer import Tokenizer

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

langs = os.listdir(args.input)
avail_langs = set()
max_idx = 0
for lang in langs:
    lang, idxext = lang.split('-')
    idx, ext = idxext.split('.')
    avail_langs.add(lang)
    max_idx = max(int(idx), max_idx)

avail_langs = sorted(list(avail_langs))
avail_langs = list(avail_langs)
required = ['be', 'en', 'hi', 'ma', 'ta', 'te', 'ur']
canon    = ['bn', 'en', 'hi', 'ml', 'ta', 'te', 'ur']
mapping = dict(zip(required, canon))
avail_langs = required


class Corpus:
    def __init__(self, path, lang):
        self.lang = lang
        with open(path) as fp:
            self.path = path
            content = fp.read()
            lines = content.splitlines()
            tokenizer = Tokenizer(lang=self.lang)
            self.lines = [tokenizer.tokenize(line) for line in lines if line.strip()]
            self.length = len(self.lines)

    def __repr__(self):
        return 'Corpus(path={}, size={})'.format(self.path, self.length)

class ApproxParallelCorpus:
    def __init__(self, paths, langs):
        self.corpus = {}
        self.paths = paths
        for path, lang in zip(self.paths, langs):
            corpus[lang] = Corpus(path, lang)

    def __repr__(self):
        _reprs = ['[']
        for _corpus in self.corpus:
            corpus = self.corpus[_corpus]
            _reprs.append('\t' + corpus.__repr__())
        _reprs.append(']')
        return '\n'.join(_reprs)

    def _run_fast_align(self):
        pivot = 'en'
        for other in self.corpus:
            if corpus.lang != 
            fpath = '/tmp/fast-align-{}-{}'.format(pivot, other):
        with open(fname)


# print(avail_langs, max_idx)
for idx in range(max_idx+1):
    paths = []
    langs = []
    for lang in avail_langs:
        path = os.path.join(args.input, '{}-{}.txt'.format(lang, idx))
        paths.append(path)
        langs.append(mapping[lang])
    parallel = ApproxParallelCorpus(paths, langs)
    print(parallel)
    exit()
