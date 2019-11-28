import os
import glob
from collections import defaultdict
import langid
from ilmulti.segment import Segmenter
from ilmulti.sentencepiece import SentencePieceTokenizer
reqs = ['hi', 'ml', 'ta', 'ur', 'te', 'bn',  'mr', 'gj', 'or']

files = defaultdict(list)
'''
def create_dir(out_dir):
    path = 'data/{}'.format(out_dir)
    try:
        os.makedirs(path)
    except OSError as error: 
        pass

for filepath in glob.iglob('*.txt'):
    idx = filepath.split('.')[0]
    lang, idx = idx.split('-')
    files[idx].append(filepath)

for fil in files:
    for path in files[fil]:
        new_path = 'data/{}/{}'.format(fil,path)
        os.rename(path, new_path)


for filepath in glob.iglob('data/*/*.txt'): 
    file = filepath.split('/')[2]
    act_lang = file.split('-')[0]
    content = open(filepath,'r')
    filt = open('{}.{}'.format(filepath,'filt'),'w+')
    for lines in content:
        lines = lines.strip()
        if lines:
            print(lines,file=filt)
            

def relaxed_canonicalize(langname):
    mapping = {'gj': 'gu', 'mp': 'bn'}
    return mapping.get(langname, langname)

for filepath in glob.iglob('data/*/*.txt.filt'): 
    file = filepath.split('/')[2]
    act_lang = file.split('-')[0]
    #print(act_lang)
    content = open(filepath,'r')
    match = open('{}.{}'.format(filepath,'match'),'w+')
    for line in content:
        pred_lang, prob = langid.classify(line)
        cpred_lang = relaxed_canonicalize(pred_lang)
        cact_lang = relaxed_canonicalize(act_lang)
        if cpred_lang == cact_lang:
            print(line,file=match)




segmenter = Segmenter()
tokenizer = SentencePieceTokenizer()

for filepath in glob.iglob('data/*/*.txt'):
    file = filepath.split('/')[2]
    lang = file.split('-')[0]
    lang = relaxed_canonicalize(lang)
    content = open(filepath,'r')
    segd = open('{}.{}'.format(filepath,'segd'),'w+') 
    out = ''
    for line in content:
        lang, segments = segmenter(line, lang=lang)
        for segment in segments:
            if segment:
                out='\n'.join(segment)
    print(out,file=segd)

def relaxed_canonicalize(langname):
    mapping = {'gj': 'gu', 'mp': 'bn'}
    return mapping.get(langname, langname)

for filepath in glob.iglob('data/*/*.txt'):
    file = filepath.split('/')[2]
    lang = file.split('-')[0]
    idx = file.split('-')[1]
    idx = idx.split('.')[0]
    lang = relaxed_canonicalize(lang)
    path = filepath
    new_path = 'data/{}/{}.{}.txt'.format(idx,idx,lang)
    os.rename(path, new_path)
'''
from ilmulti.utils.language_utils import inject_token
from io import StringIO
from bleualign.align import Aligner
from align import BLEUAligner
from ilmulti.translator.pretrained import mm_all
from tqdm import tqdm

segmenter = Segmenter()
tokenizer = SentencePieceTokenizer()
root = '/home/darth.vader/.ilmulti/mm-all'
translator = mm_all(root=root, use_cuda=True).get_translator()
aligner = BLEUAligner(translator, tokenizer, segmenter)


def relaxed_canonicalize(langname):
    mapping = {'gj': 'gu', 'mp': 'bn'}
    return mapping.get(langname, langname)

def create_stringio(lines, lang):
    tokenized = [ ' '.join(tokenizer(line, lang=lang)[1]) \
            for line in lines ]
    lstring = '\n'.join(tokenized)
    return tokenized, StringIO(lstring)

def detok(src_out):
    src = []
    for line in src_out:
        src_detok = tokenizer.detokenize(line)
        src.append(src_detok)
    return src   

def prepare(content, src_lang, tgt_lang):
    tok, _io = create_stringio(content, src_lang)
    injected = inject_token(tok, tgt_lang)
    hyp = translator(injected)
    hyp = [ gout['tgt'] for gout in hyp]
    hyp = '\n'.join(hyp)
    hyp_io = StringIO(hyp)

    return _io, hyp_io


for filepath in tqdm(glob.iglob('data/*/*.txt')):
    file = filepath.split('/')[2]
    lang = file.split('.')[1]
    idx = file.split('.')[0]
    lang = relaxed_canonicalize(lang)
    src_lang, tgt_lang = lang, 'en' 
    src_content = open(filepath,'r')
    tgt_content = open('data/{}/{}.{}.txt'.format(idx, idx, tgt_lang),'r')

    src_io, src_tgt_io = prepare(src_content, src_lang, tgt_lang)
    tgt_io, tgt_src_io = prepare(tgt_content, tgt_lang, src_lang)

    src_aligned, tgt_aligned = aligner.bleu_align(src_io, tgt_io, src_tgt_io, tgt_src_io)

    src_aligned = detok(src_aligned)
    tgt_aligned = detok(tgt_aligned)               
    src_entry = '\n'.join(src_aligned)
    tgt_entry = '\n'.join(tgt_aligned)
    
    src_file = open('data/{}/{}.{}-{}.{}.alg.txt'.format(idx,idx,src_lang, tgt_lang, src_lang),'a')
    tgt_file = open('data/{}/{}.{}-{}.{}.alg.txt'.format(idx,idx,src_lang, tgt_lang, tgt_lang),'a')
    print(src_entry, file=src_file)
    print(tgt_entry, file=tgt_file)
