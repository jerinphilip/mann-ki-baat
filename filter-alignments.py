import sys
sys.path.insert(1, '../')
from io import StringIO
from ilmulti.segment import Segmenter
from ilmulti.sentencepiece import SentencePieceTokenizer
from tqdm import tqdm
from argparse import ArgumentParser
import langid

from langid.langid import LanguageIdentifier, model
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
from generate_multi import ParallelWriter


def eval_len_ratio(src_len, tgt_len):
    # if src_len==0 or tgt_len==0:
    #     return False
    # ratio = src_len/tgt_len
    # src = (src_len >=2)#(2 <= src_len <= 50)
    # tgt = (tgt_len >=2)#(2 <= tgt_len <= 50)
    # if 0.5 <= ratio <= 2 and src and tgt:
        return True

def eval_lang(src_lang, src_line, tgt_lang, tgt_line):
    threshold = 0.9
    slang, src_prob = identifier.classify(src_line)
    tlang, tgt_prob = identifier.classify(tgt_line)
    src = (src_prob >= threshold)
    tgt = (tgt_prob >= threshold)
    if slang==src_lang and tlang==tgt_lang and src and tgt:
        return True
    else:
        return False

def filter_lines(src_lang, src_file, tgt_lang, tgt_file):
    src_uniq = set()
    tgt_uniq = set()
    for src_line, tgt_line in zip(src_file, tgt_file):
        src_len = len(src_line.split())
        tgt_len = len(tgt_line.split())
        len_eval = eval_len_ratio(src_len, tgt_len)
        lang_eval = eval_lang(src_lang, src_line, tgt_lang, tgt_line)
        if len_eval and lang_eval :
            src_line = src_line.strip('\n')
            tgt_line = tgt_line.strip('\n')
            if src_line not in src_uniq and tgt_line not in tgt_uniq:
                src_uniq.add(src_line)
                tgt_uniq.add(tgt_line)
                pwriter.write(src_lang, tgt_lang, src_line,  tgt_line)



if __name__ == '__main__':

    reqs = ['hi', 'ml', 'ta', 'ur', 'te','bn','mr', 'gu', 'or']
    fpath = './mkb-filt/'
    fname = 'mkb'
    pwriter = ParallelWriter(fpath, fname)
    for lang in reqs:
        def dirname(xx):
            fst, snd = sorted([xx, 'en'])
            return '{}-{}'.format(fst, snd)
        dxx = dirname(lang)
        src_file = open('./mkb/{}/mkb.{}'.format(dxx, lang),'r')
        tgt_file = open('./mkb/{}/mkb.{}'.format(dxx, 'en'),'r')
        identifier.set_languages(['{}'.format(lang),'{}'.format('en')])
        filter_lines(lang, src_file, 'en', tgt_file)

