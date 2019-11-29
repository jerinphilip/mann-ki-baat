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
        # print("Segments({}) = {}, ({}, {}, {})".format(src_lang, len(segments),
        #     np.min(lengths), np.mean(lengths), np.max(lengths)
        #     ))
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
        return tokenized, StringIO(lstring), lengths


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
                    print("Check fpath:", fpath)
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
                    tgt_io, src_tgt_io, tgt_src_io)

            src_aligned = self.postprocess(src_aligned)
            tgt_aligned = self.postprocess(tgt_aligned)

            key = tuple(sorted([self.anchor, other_lang]))
            aligned[key] = {
                self.anchor : src_aligned,
                other_lang: tgt_aligned
            }

        return aligned

if __name__ == '__main__':
    segmenter = Segmenter()
    tokenizer = SentencePieceTokenizer()
    root = '/home/darth.vader/.ilmulti/mm-all'
    translator = mm_all(root=root, use_cuda=True).get_translator()
    aligner = BLEUAligner(translator, tokenizer, segmenter)
    langs = ['en', 'hi', 'ml', 'ta', 'ur', 'te', 'bn',  'mr', 'gu', 'or']
    # langs = ['hi', 'en', 'ml']
    speech_translator = SpeechTranslator(segmenter, tokenizer, translator)
    mkb_storage_dir=sys.argv[1]
    # anchor_lang = 'en'
    anchor_lang = 'en'
    MKB = OrganizeMKB(mkb_storage_dir, langs, anchor_lang, speech_translator)
    for idx in range(57):
        speech = MKB.collect(idx)
        # print(speech)
        aligned = MKB.align(speech)
        for key in aligned:
            print(key, len(aligned[key][anchor_lang]))
