import requests
from lxml import html, etree
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument('--output', help='dir to output', type=str,
        required=True)
args = parser.parse_args()

required = ['be', 'en', 'hi', 'ma', 'ta', 'te', 'ur']
canon    = ['bn', 'en', 'hi', 'ml', 'ta', 'te', 'ur']
mapping = dict(zip(required, canon))

if not os.path.exists(args.output):
    os.mkdir(args.output)

def save(lang, idx, text):
    lang = mapping[lang]
    fname = '{}-{}.txt'.format(lang, idx)
    fpath = os.path.join(args.output, fname)
    with open(fpath, 'w+') as fp:
        print(text, file=fp)

def access_langs(tree):
    urls = tree.xpath('//ul[@id="access_lang_items"]//a/@href')
    prefixes = []
    for url in urls:
        prefix = url.replace("https://www.narendramodi.in/", "")
        prefix = prefix.replace("mann-ki-baat", "")
        prefix = prefix.replace("/", "")
        if not prefix:
            prefix='en'
        prefixes.append(prefix)
    return urls, prefixes

page = requests.get("https://www.narendramodi.in/hi/mann-ki-baat")
tree = html.fromstring(page.content)
urls, prefixes = access_langs(tree)

for url, prefix in zip(urls, prefixes):
    page = requests.get(url)
    tree = html.fromstring(page.content)
    mcsbs = tree.xpath('//div[@class="readMoreDv"]')
    for i, mcsb in enumerate(mcsbs):
        print(i, mcsb.text_content())
        save(prefix, i, mcsb.text_content().decode("utf-8"))

