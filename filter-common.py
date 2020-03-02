import os
import glob
from collections import defaultdict
import langid
from generate_multi import ParallelWriter
#from ilmulti.segment import Segmenter
#from ilmulti.sentencepiece import SentencePieceTokenizer

reqs = ['hi', 'ml', 'ta', 'ur', 'te','bn','mr', 'gu', 'or']
for lang in reqs:
        mkb = defaultdict(list)
        pib = defaultdict(list)
        fpath = './pib-filt/'
        fname = 'train'
        pwriter = ParallelWriter(fpath, fname)
        def dirname(xx):
            fst, snd = sorted([xx, 'en'])
            return '{}-{}'.format(fst, snd)

        ldxx = dirname(lang)

        with open('./mkb/{}/mkb.{}'.format(dxx, lang)) as src1,\
             open('./mkb/{}/mkb.en'.format(dxx, lang)) as tgt1,\
             open('./mkb/{}/pib.en-{}.{}.filt.txt'.format(dxx, lang, lang)) as src2,\
             open('./mkb/{}/pib.en-{}.en.filt.txt'.format(dxx, lang)) as tgt2:
            for s1, t1 in zip(src1, tgt1):
                s1 = s1.strip()
                t1 = t1.strip()
                mkb[s1].append(t1)
            smkb = set(mkb)
            for s2, t2 in zip(src2, tgt2):
                s2 = s2.strip()
                t2 = t2.strip()
                pib[s2].append(t2)
            spib = set(pib)

            for k in spib.difference(smkb):
                pwriter.write(lang, 'en', k, pib[k][0])
                


