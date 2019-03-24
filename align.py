from __future__ import unicode_literals
import re
from argparse import ArgumentParser
import os
from polyglot_tokenizer import Tokenizer
import subprocess as sp
import numpy as np
import requests
import json
from bleualign.align import Aligner
from collections import defaultdict
from tqdm import tqdm
from pf.segment import Segmenter
from pf.sentencepiece import SentencePieceTokenizer
import pf
import io
from collections import namedtuple


tokenizer = SentencePieceTokenizer()
segment = Segmenter()

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--mapping', type=str, required=True)
parser.add_argument('--fseqout', type=str, required=True)
falign_executable = '/home/jerin/builds/fast_align/build'
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
    def __init__(self, lines, path='in-memory', lang=None):
        self.lines = lines
        self.length = len(lines)
        self.lang = lang
        self.path = path

    @classmethod
    def build(cls, path, lang):
        with open(path) as fp:
            content = fp.read()
            lang, lines = segment(content, lang=lang)
        return cls(lines, path, lang)

    def vocabulary(self):
        vocab = set()
        for line in self.lines:
            v = set(line)
            vocab = vocab.union(v)

        return sorted(list(vocab))

    def __add__(self, other):
        return Corpus(self.lines + other.lines)

    def __repr__(self):
        return 'Corpus(path={}, size={})'.format(self.path, self.length)


# class ApproxParallelCorpus:
#     def __init__(self, corpus):
#         self.corpus = corpus
# 
#     @classmethod
#     def build(cls, paths, langs):
#         corpus = {}
#         for path, lang in zip(paths, langs):
#             corpus[lang] = Corpus.build(path, lang)
#         return cls(corpus)
# 
#     def __add__(self, other):
#         _new = {}
#         for key in self.corpus:
#             _new[key] = self.corpus[key] + other.corpus[key]
#         return ApproxParallelCorpus(_new)
# 
# 
#     def __repr__(self):
#         _reprs = ['[']
#         for _corpus in self.corpus:
#             corpus = self.corpus[_corpus]
#             _reprs.append('\t' + corpus.__repr__())
#         _reprs.append(']')
#         return '\n'.join(_reprs)
#     
# 
# def translate(sample, to, _from='-detect-'):
#     args = {
#         "src_lang": _from,
#         "tgt_lang": to,
#         "content": sample,
#         "system": "mm-v1"
#     }
# 
#     url = 'http://preon.iiit.ac.in/babel/api/'
#     response = requests.post(url, args)
#     jstring = response.content.decode("utf-8")
#     package = json.loads(jstring)
#     # print(sample, package)
#     try:
#         translation = package[0]['hypotheses'][0]['prediction_raw']
#     except:
#         translation = ''
# 
#     return translation


def align(srcfile, tgtfile, approx_src_tgt_file):
    srcfile, tgtfile = tgtfile, srcfile
    options = {
        # source and target files needed by Aligner
        # they can be filenames, arrays of strings or io objects.
        'srcfile': srcfile,
        'targetfile': tgtfile,
        # translations of srcfile and targetfile, not influenced by 'factored'
        # they can be filenames, arrays of strings or io objects, too.
        'srctotarget': [approx_src_tgt_file],
        'targettosrc': [],
        # passing filenames or io object for them in respectly.
        # if not passing anything or assigning None, they will use StringIO to save results.
        'output-src': None, 
        'output-target': None,
        # other options ...
    }
    a = Aligner(options)
    a.mainloop()
    output_src, output_target = a.results()
    # output_src, output_target is StringIO because options['output-src'] is None
    src = output_src.getvalue().splitlines()  # StringIO member function
    tgt = output_target.getvalue().splitlines()  # array of string
    return (src, tgt)


class FSeqReal:
    def __init__(self, mapping_file, save_dir):
        self.build_mapping(mapping_file)

    def build_mapping(self, mapping_file):
        mapping = defaultdict(list)
        with open(mapping_file) as fp:
            for line in fp:
                tag, idx = line.split()
                mapping[idx] = tag
                # mapping[(ssrc, stgt)].append(sid)
        self.mapping = mapping

    def _process_one(self, txt, is_hyp=False):
        entries = txt.split('\t')
        Id = entries[0][2:]
        # print(entries)
        value = entries[2] if is_hyp else entries[1]
        value = value.replace(' ', '')
        value = value.replace('▁', ' ')
        value = value[1:]
        return (Id, value)

    def process(self, outfile):
        export = {}
        srcs = {}
        tgts = {}
        hyps = {}
        with open(outfile) as fp:
            for i, line in enumerate(fp):
                if i > 4:
                    break

            completed = False
            while not completed:
                try:
                    src = next(fp).strip()
                    Id, value = self._process_one(src)
                    srcs[Id] = value

                    tgt = next(fp).strip()
                    Id, value = self._process_one(tgt)
                    tgts[Id] = value

                    hyp = next(fp).strip()
                    Id, value = self._process_one(hyp, is_hyp=True)
                    hyps[Id] = value
                    _ = next(fp)
                except IndexError:
                    pass
                except StopIteration:
                    completed = True
            for idx in self.mapping:
                if idx in srcs and idx in hyps:
                    tag = self.mapping[idx]
                    export[tag] = {
                        "src": srcs[idx],
                        "tgt": tgts[idx]
                    }
        return export


parallel = None
idy = 0

fseq = FSeqReal(args.mapping, None)
export = fseq.process(args.fseqout)

class Writer:
    def __init__(self, tag, srcs, tgts, hyps):
        self.tag = tag
        self.srcs = srcs
        self.hyps = hyps
        self.tgts = tgts

    def io(self):
        srcs = '\n'.join(self.srcs)
        src = io.StringIO(srcs)

        tgts = '\n'.join(self.tgts)
        tgt = io.StringIO(tgts)

        hyps = '\n'.join(self.hyps)
        hyp = io.StringIO(hyps)
        return src, hyp, tgt


    @classmethod
    def build(cls, _export, idx, src_lang, tgt_lang='en'):
        path = os.path.join(args.input, '{}-{}.txt'.format(src_lang, idx))
        src_corpus = Corpus.build(path, src_lang)

        path = os.path.join(args.input, '{}-{}.txt'.format(tgt_lang, idx))
        tgt_corpus = Corpus.build(path, tgt_lang)

        hyps = {}

        for i, line in enumerate(src_corpus.lines):
            tag = '{}.{}.{}'.format(lang, idx, i)
            # lang, tokens = tokenizer(line, lang=src_lang)
            # sequence = ' '.join(tokens)
            # sample = '{} {}'.format(
            #     pf.utils.language_token('en'), 
            #     sequence
            # )
            # exit
            # print(tag, idy, file=bijection)
            # print(sample, file=outfile)
            if tag in export:
                hyps[i] = _export[tag]["tgt"]
                src = _export[tag]["src"][8:]
                # condition = line == src
                # if not condition:
                #     print(line, src)
            else:
                hyps[i] = ''

        _hyps = []
        for i in sorted(hyps.keys()):
            _hyps.append(hyps[i])


        tag = '{}.{}'.format(idx, lang)
        instance =  cls(tag, src_corpus.lines, tgt_corpus.lines, _hyps)
        return instance


langwise = defaultdict(list)


for idx in range(max_idx+1):
    for lang in avail_langs:
        if lang != 'en':
            writer = Writer.build(export, idx, lang)
            src, hyp, tgt = writer.io()
            asrc, atgt = align(src, tgt, hyp)
            for _asrc, _atgt in zip(asrc, atgt):
                langwise[(lang, 'en')].append((_asrc, _atgt))

    if max_idx > 8:
        break

for src_lang, tgt_lang in langwise:
    src_lang = 'ma'
    tgt_lang = 'en'
    ls = langwise[(src_lang, tgt_lang)]
    print(ls)
    for _asrc, _atgt in ls:
        print('>', _asrc)
        print('<', _atgt)

    break
        
    
    # paths = []
    # langs = []
    # for lang in avail_langs:
    #     path = os.path.join(args.input, '{}-{}.txt'.format(lang, idx))
    #     paths.append(path)
    #     langs.append(mapping[lang])

    # for path, lang in zip(paths, langs):
    #     if lang != 'en':
                
    # _parallel = ApproxParallelCorpus.build(paths, langs)
    # pivot = 'en'
    # for lang in langs:
    #     if lang != pivot:
    #         left = _parallel.corpus[lang]
    #         right = _parallel.corpus[pivot]
    #         aligner = Aligner(left, right)

