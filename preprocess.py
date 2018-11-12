import json

import re

import nltk
import enchant

from util import map_pos


lemmatizer = nltk.WordNetLemmatizer()
checker = enchant.Dict('en_US')


def prepare(path_txt, path_json, detail):
    sents = dict()
    quaples = list()
    errors = dict()
    with open(path_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not re.findall('-DOCSTART-', line):
                quaple = dict()
                word, pos, chunk, label = line.split()
                word = lemmatizer.lemmatize(word, map_pos(pos))
                if detail and word.isalpha() and word.islower() and not checker.check(word):
                    errors[word] = checker.suggest(word)
                quaple['word'] = word
                quaple['pos'] = pos
                quaple['chunk'] = chunk
                quaple['label'] = label
                quaples.append(quaple)
            elif quaples:
                text = ' '.join([quaple['word'] for quaple in quaples])
                sents[text] = quaples
                quaples = []
    with open(path_json, 'w') as f:
        json.dump(sents, f, ensure_ascii=False, indent=4)
    if errors:
        for word, cands in errors.items():
            print('%s -> (%s)' % (word, ', '.join(cands)))


if __name__ == '__main__':
    path_txt = 'data/train.txt'
    path_json = 'data/train.json'
    prepare(path_txt, path_json, detail=False)
    path_txt = 'data/dev.txt'
    path_json = 'data/dev.json'
    prepare(path_txt, path_json, detail=False)
    path_txt = 'data/test.txt'
    path_json = 'data/test.json'
    prepare(path_txt, path_json, detail=False)
