import os
import sys
import langid
from collections import defaultdict
from ilmulti.segment import Segmenter
from ilmulti.sentencepiece import SentencePieceTokenizer
from argparse import ArgumentParser
from ilmulti.utils.language_utils import inject_token
from io import StringIO
from bleualign.align import Aligner
from align import BLEUAligner
from ilmulti.translator.pretrained import mm_all
from tqdm import tqdm
from copy import deepcopy
import numpy as np

class SpeechTranslator:
    def __init__(self, segmenter, tokenizer, translator):
        self.segmenter = segmenter
        self.tokenizer = tokenizer
        self.translator = translator

    def __call__(self, content, src_lang, tgt_lang):
        _lang, segments = self.segmenter(content, lang=src_lang)
        tok, _io, lengths = self.create_stringio(segments, src_lang)
        injected = inject_token(tok, tgt_lang)
        hyp = self.translator(injected)
        hyp = [ gout['tgt'] for gout in hyp]
        hyp = '\n'.join(hyp)
        hyp_io = StringIO(hyp)
        return _io, hyp_io

    def create_stringio(self, lines, lang):
        tokenized_lines = []
        lengths = []
        for line in lines:
            lang, tokenized = self.tokenizer(line, lang=lang)
            lengths.append(len(tokenized))
            tokenized = ' '.join(tokenized)
            tokenized_lines.append(tokenized)
        lstring = '\n'.join(tokenized_lines)
        return tokenized_lines, StringIO(lstring), lengths


class OrganizeMKB:
    def __init__(self, mkb_storage_dir, langs, anchor, speech_translator):
        self.dir = mkb_storage_dir
        self.langs = langs
        self.anchor = anchor
        self.speech_translator = speech_translator

    def preprocess(self, speech_content):
        cleaned = []
        lines = speech_content.splitlines()
        for line in lines:
            clean = line.strip()
            # Some content should exist.
            if clean:
                cleaned.append(clean)
        return '\n'.join(cleaned)

    def fname_hack(self, lang):
        mappings = {
            "gu": "gj",
        }
        return mappings.get(lang, lang)

    def collect(self, idx):
        speech = {}
        for lang in self.langs:
            _lang = self.fname_hack(lang)
            fname = '{}-{}.txt'.format(_lang, idx)
            fpath = os.path.join(self.dir, fname)
            with open(fpath) as fp:
                speech_content = fp.read()
                # TODO(shashank) Sanitize with langid or whatever other
                # Only allow matching stuff to go in.
                if self._filter(lang, speech_content):
                    speech[lang] = self.preprocess(speech_content)
                else:
                    pass
                    # print("Check fpath:", fpath)
        return speech

    def _filter(self, lang, content):
        langid.set_languages(self.langs)
        predicted_lang, _ = langid.classify(content)
        if predicted_lang != lang:
            return False
        return True

    def postprocess(self, lines):
        processed = []
        for line in lines:
            processed_line = self.speech_translator.tokenizer.detokenize(line)
            processed.append(processed_line)
        return processed

    def align(self, speech):
        if self.anchor not in speech:
            return {}
        anchor_content = deepcopy(speech[self.anchor])
        del speech[self.anchor]

        translated = {}
        aligned = {}
        for other_lang in speech:
            other_content = speech[other_lang]
            src_io, src_tgt_io = self.speech_translator(other_content, 
                    src_lang=other_lang, tgt_lang=self.anchor)

            tgt_io, tgt_src_io = self.speech_translator(anchor_content, 
                    src_lang=self.anchor, tgt_lang=other_lang)

            src_aligned, tgt_aligned = aligner.bleu_align(src_io, 
                    tgt_io, src_tgt_io,
                    tgt_src_io
            )


            src_aligned = self.postprocess(src_aligned)
            tgt_aligned = self.postprocess(tgt_aligned)

            assert(len(src_aligned) == len(tgt_aligned))

            key = tuple([self.anchor, other_lang])
            aligned[key] = {
                self.anchor : tgt_aligned,
                other_lang: src_aligned
            }

        return aligned

    def process(self, count, path):
        collected = defaultdict(list)
        for idx in range(count):
            print("Article", idx)
            speech = self.collect(idx)
            aligned = self.align(speech)
            for key in aligned:
                anchor, other = key
                for anchor_sentence, other_sentence in zip(aligned[key][anchor], aligned[key][other]):
                    # print(anchor, anchor_sentence)
                    # print(other, other_sentence)
                    collected[anchor_sentence].append(
                        (other, other_sentence)
                    )
                    pkey = tuple(sorted([anchor, other]))

        scattered = defaultdict(list)
        from itertools import combinations
        for anchor_sentence in collected:
            ls = [(self.anchor, anchor_sentence)] + collected[anchor_sentence]
            samples = combinations(ls, 2)
            for ps in samples:
                (xx, xx_sentence), (yy, yy_sentence) = sorted(ps, key=lambda x: x[0])
                scattered[(xx, yy)].append(
                    (xx_sentence, yy_sentence)
                )

        for pkey in scattered:
            xx, yy = pkey
            fxx = os.path.join(path, '{}-{}.{}'.format(xx, yy, xx))
            fyy = os.path.join(path, '{}-{}.{}'.format(xx, yy, yy))
            with open(fxx, 'w+') as fpxx, open(fyy, 'w+') as fpyy:
                samples = list(sorted(scattered[pkey]))
                for sample in samples:
                    sxx, syy = sample
                    print(sxx, file=fpxx)
                    print(syy, file=fpyy)
        
if __name__ == '__main__':
    mkb_storage_dir = sys.argv[1]
    output_dir = sys.argv[2]

    segmenter = Segmenter()
    tokenizer = SentencePieceTokenizer()
    root = '/home/darth.vader/.ilmulti/mm-all'
    translator = mm_all(root=root, use_cuda=True).get_translator()
    aligner = BLEUAligner(translator, tokenizer, segmenter)
    langs = ['en', 'hi', 'ml', 'ta', 'ur', 'te', 'bn',  'mr', 'gu', 'or']
    # langs = ['en', 'hi', 'ml']
    speech_translator = SpeechTranslator(segmenter, tokenizer, translator)
    anchor_lang = 'en'
    MKB = OrganizeMKB(mkb_storage_dir, langs, anchor_lang, speech_translator)
    MKB.process(58, output_dir)

