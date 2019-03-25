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
from tqdm import tqdm, trange
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
# avail_langs = required


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
            _lang, lines = segment(content, lang=lang)
            if lang != _lang:
                lines = []
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


def align(srcfile, tgtfile, approx_src_tgt_file):
    # srcfile, tgtfile = tgtfile, srcfile
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
        'log_to': open("/dev/null", "w+")
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
                        "tgt": hyps[idx]
                    }
        return export


parallel = None
idy = 0

fseq = FSeqReal(args.mapping, None)
export = fseq.process(args.fseqout)

class Writer:
    def __init__(self, tag, srcs, tgts, hyps):
        self.tag = tag
        self.srcs = self.clean(srcs)
        self.hyps = self.clean(hyps)
        self.tgts = self.clean(tgts)
        # exit()
        

    def clean(self, lines):
        ls = []
        for line in lines:
            line = line.lstrip().rstrip()
            line = re.sub('\s\s+', ' ', line)
            if line:
                ls.append(line)
        return ls

    def io(self):
        srcs = '\n'.join(self.srcs)
        src = io.StringIO(srcs)

        tgts = '\n'.join(self.tgts)
        tgt = io.StringIO(tgts)

        hyps = '\n'.join(self.hyps)
        hyp = io.StringIO(hyps)
        return src, hyp, tgt

    def files(self, prefix):
        def __inner(ext, lines):
            fname = prefix + ext
            with open(fname, 'w+') as fp:
                for line in lines:
                    print(line, file=fp)
            return fname
        src = __inner(".src", self.srcs)
        tgt = __inner(".tgt", self.tgts)
        hyp = __inner(".hyp", self.hyps)
        return src, hyp, tgt


    @classmethod
    def build(cls, _export, idx, src_lines, tgt_lines, src_lang, tgt_lang='en'):
        hyps = {}
        for i, line in enumerate(src_lines):
            tag = '{}.{}.{}'.format(src_lang, idx, i)
            if tag in export:
                hyps[i] = _export[tag]["tgt"]
                src = _export[tag]["src"][8:]
                # if src_lang in ['ml' or 'hi']:
                    # print('>', line)
                    # print('<', src)
                    # print('=', hyps[i])
                # condition = line == src
                # if not condition:
                #     print(line, src)
            else:
                hyps[i] = ''

        _hyps = []
        for i in sorted(hyps.keys()):
            _hyps.append(hyps[i])


        tag = '{}.{}'.format(idx, lang)
        instance =  cls(tag, src_lines, tgt_lines, _hyps)
        return instance



def create(mapping_file, outfile):
    fp =  open(mapping_file, 'w+')
    outfile =  open(outfile, "w+")

    counter = 0
    for idx in range(max_idx+1):
        for lang in avail_langs:
            if lang != 'en':
                fname = '{}-{}.txt'.format(lang, idx)
                path = os.path.join(args.input, fname)
                src = Corpus.build(path, lang).lines
                for i, line in enumerate(src):
                    tag = '{}.{}.{}'.format(lang, idx, i)
                    tag = '{}.{}.{}'.format(lang, idx, i)
                    lang, tokens = tokenizer(line, lang=lang)
                    sequence = ' '.join(tokens)
                    sample = '{} {}'.format(
                        pf.utils.language_token('en'), 
                        sequence
                    )
                    print(tag, counter, file=fp)
                    print(sample, file=outfile)
                    counter += 1


langwise = defaultdict(list)

# trange = lambda x: range(x)
# tqdm = lambda x: x
for idx in trange(max_idx+1):
    for lang in tqdm(avail_langs):
        if lang != 'en':
            src_lang = lang
            tgt_lang = 'en'
            path = os.path.join(args.input, '{}-{}.txt'.format(src_lang, idx))
            src_lines = Corpus.build(path, src_lang).lines

            path = os.path.join(args.input, '{}-{}.txt'.format(tgt_lang, idx))
            tgt_lines = Corpus.build(path, tgt_lang).lines

            if src_lines:
                writer = Writer.build(export, idx, src_lines, tgt_lines, src_lang, tgt_lang)
                # src, hyp, tgt = writer.io()
                tag = '{}-{}.{}'.format(src_lang, tgt_lang, idx)
                fprefix = os.path.join(args.output, tag)
                src, hyp, tgt = writer.files(fprefix)
                asrc, atgt = align(src, tgt, hyp)

                for _asrc, _atgt in zip(asrc, atgt):
                    langwise[(src_lang, 'en')].append((_asrc, _atgt))
                #     if src_lang in ['ml', 'hi']:
                #         print('>', _asrc)
                #         print('<', _atgt)


_dirpath = os.path.join(args.output, 'langwise') 
if not os.path.exists(_dirpath):
    os.makedirs(_dirpath)

fps = {}
for src_lang, tgt_lang in langwise:
    fprefix = os.path.join(_dirpath, '{}-{}'.format(src_lang,
        tgt_lang))
    fps[(src_lang, tgt_lang)] = {
        "src": open(fprefix + ".src", "w+"),
        "tgt": open(fprefix + ".tgt", "w+")
    }

for src_lang, tgt_lang in langwise:
    key = src_lang, tgt_lang
    ls = langwise[key]
    for _asrc, _atgt in ls:
        print(_asrc, file=fps[key]["src"])
        print(_atgt, file=fps[key]["tgt"])
