import requests
from lxml import html, etree
from argparse import ArgumentParser
import os

cookies = {
    'BJP-LB': 'WEB1',
    'bm_mi': 'FADB312AF7B6CB647731F9442DFA8003~w2As1V1+jsYfYXpL9q1xGCzG4IS8DYcdf/AIK1+KhMaFyOeICADIAVQG+Nb6eyeibpns+we4Uws4IVB8T3ppwmjuz+BX7pfchZvh9fV+EgpbA3Au+aGTNB10UBNus49IlWATomua1gHXD8vuhuZcfMo5IU/PmotMx0O4KD7vFzsNLWzQfU3ti+PIbBXFc8LJgao2SwRk+d1ocztY73V3XLwCrjE3l3DjpCvzNE4IBL6nSgMI7I0JIil5I0JYXrRa',
    'bm_sv': 'FFBE93FF1AB62087468C1FB311759DDE~nDauqABPSZ8exWGhtZ2SGD5ZiLV/iQzsi4XoZzHq2+zUYCc1mR9PClJkfuCr9MFx95bgx7svVVldZKAw3hD0+HRVj4Y45njkK/RkHbKUVKd/s5FCLcm06nJNarlmGPUWKbnt/cvjRq7ttzfQ3ZWzE6xzH6QScNPjoPWX49swqsk=',
    'ak_bmsc': '188973CE7ECBB004C2A604E980FF3BDF7C7CFC9F5761000035B9D25D1773031C~pldt8gejnDgep+ci6cFa6EFa3/Fn1iSRAElmZTNcbbsSW8KdQdaPerWXTiq7i9KkxcXw/2iWL/vYroDFeiVJP0NpTYxo87cksrUPMz70Bh4OTGJviL0sFc7YzCcAwxDIEWnmkPXCDpspL3f8ueOCaRhfaN0Cd4VotRFUjXax49zxQyqkGVvc8VuKKJru73kWm4sn9c00WuZyVRfhwgoxkroBBHXev3F3E/jKF1XIOx/uyF6+HVhegsurT6tIKyylpmlA1a5tXHUvB+mndCB9QFZw==',
    '_webUser': 'l%2C+d-M-Y+H%3Ai%3As+T',
    'WZRK_S_4RR-W49-K84Z': '%7B%22p%22%3A12%7D',
}

headers = {
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-User': '?1',
    'DNT': '1',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Sec-Fetch-Site': 'none',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
}

# response = requests.get('https://www.narendramodi.in/ka/mann-ki-baat', headers=headers, cookies=cookies)



parser = ArgumentParser()
parser.add_argument('--output', help='dir to output', type=str,
        required=True)
args = parser.parse_args()

required = ['be', 'en', 'hi', 'ma', 'mar', 'ta', 'te', 'ur', 'asm', 'gu', 'ka', 'od',  'man']
canon    = ['bn', 'en', 'hi', 'ml', 'mr', 'ta', 'te', 'ur', 'as', 'gj', 'kn', 'or', 'mp']
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

page = requests.get("https://www.narendramodi.in/hi/mann-ki-baat",  headers=headers, cookies=cookies)

tree = html.fromstring(page.content)
urls, prefixes = access_langs(tree)


for url, prefix in zip(urls, prefixes):
    page = requests.get(url, headers=headers, cookies=cookies)
    tree = html.fromstring(page.content)
    mcsbs = tree.xpath('//div[@class="readMoreDv"]')
    for i, mcsb in enumerate(mcsbs):
        print(i, url, type(mcsb.text_content()))
        save(prefix, i, mcsb.text_content())

