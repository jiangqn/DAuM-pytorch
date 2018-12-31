import os
import ast
import spacy
import numpy as np
from errno import ENOENT
from collections import Counter

nlp = spacy.load("en")

def get_data_info(dataset, pre_processed):
    train_fname = dataset + 'train.txt'
    test_fname = dataset + 'test.txt'
    save_fname = dataset + 'data_info.txt'

    word2id, max_aspect_len, max_sentence_len = {}, 0, 0
    word2id['<pad>'] = 0
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        with open(save_fname, 'r') as f:
            for line in f:
                content = line.strip().split()
                if len(content) == 3:
                    max_aspect_len = int(content[1])
                    max_sentence_len = int(content[2])
                else:
                    word2id[content[0]] = int(content[1])
    else:
        if not os.path.isfile(train_fname):
            raise IOError(ENOENT, 'Not a file', train_fname)
        if not os.path.isfile(test_fname):
            raise IOError(ENOENT, 'Not a file', test_fname)

        words = []

        lines = open(train_fname, 'r').readlines()
        for i in range(0, len(lines), 3):
            sptoks = nlp(lines[i].strip())
            words.extend([sp.text.lower() for sp in sptoks])
            context_len = len(sptoks) - 1
            sptoks = nlp(lines[i + 1].strip())
            aspect_len = len(sptoks)
            words.extend([sp.text.lower() for sp in sptoks])
            if context_len + aspect_len > max_sentence_len:
                max_sentence_len = context_len + aspect_len
            if aspect_len > max_aspect_len:
                max_aspect_len = aspect_len
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word and '\n' not in word and 'aspect_term' not in word:
                word2id[word] = len(word2id)

        lines = open(test_fname, 'r').readlines()
        for i in range(0, len(lines), 3):
            sptoks = nlp(lines[i].strip())
            words.extend([sp.text.lower() for sp in sptoks])
            context_len = len(sptoks) - 1
            sptoks = nlp(lines[i + 1].strip())
            aspect_len = len(sptoks)
            words.extend([sp.text.lower() for sp in sptoks])
            if context_len + aspect_len > max_sentence_len:
                max_sentence_len = context_len + aspect_len
            if aspect_len > max_aspect_len:
                max_aspect_len = aspect_len
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word and '\n' not in word and 'aspect_term' not in word:
                word2id[word] = len(word2id)

        with open(save_fname, 'w') as f:
            f.write('length %s %s\n' % (max_aspect_len, max_sentence_len))
            for key, value in word2id.items():
                f.write('%s %s\n' % (key, value))

    print('There are %s words in the dataset, the max length of aspect is %s, and the max length of sentence is %s' % (
    len(word2id), max_aspect_len, max_sentence_len))
    return word2id, max_aspect_len, max_sentence_len


def read_data(word2id, max_aspect_len, max_sentence_len, dataset, pre_processed):
    fname = dataset + '.txt'
    save_fname = dataset + '.npz'

    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        return save_fname
    else:
        aspects, sentences, labels, aspect_lens, sentence_lens, aspect_positions = list(), list(), list(), list(), list(), list()
        if not os.path.isfile(fname):
            raise IOError(ENOENT, 'Not a file', fname)
        lines = open(fname, 'r').readlines()
        for i in range(0, len(lines), 3):
            polarity = lines[i + 2].split()[0]
            if polarity == 'conflict':
                continue

            aspect_sptoks = nlp(lines[i + 1].strip())
            aspect = []
            for aspect_sptok in aspect_sptoks:
                if aspect_sptok.text.lower() in word2id:
                    aspect.append(word2id[aspect_sptok.text.lower()])

            context_sptoks = nlp(lines[i].strip())
            sentence = []
            aspect_position = []
            for sptok in context_sptoks:
                if sptok.text.lower() in word2id:
                    sentence.append(word2id[sptok.text.lower()])
                elif sptok.text.lower() == 'aspect_term':
                    aspect_position.extend([0] * len(sentence))
                    sentence.extend(aspect)
                    aspect_position.extend([1] * len(aspect))

            aspects.append(aspect + [0] * (max_aspect_len - len(aspect)))
            sentences.append(sentence + [0] * (max_sentence_len - len(sentence)))
            aspect_positions.append(aspect_position + [0] * (max_sentence_len - len(aspect_position)))
            if polarity == 'negative':
                labels.append(0)
            elif polarity == 'neutral':
                labels.append(1)
            elif polarity == 'positive':
                labels.append(2)
            aspect_lens.append(len(aspect))
            sentence_lens.append(len(sentence))
        print("Read %s examples from %s" % (len(aspects), fname))
        aspects = np.asarray(aspects)
        sentences = np.asarray(sentences)
        labels = np.asarray(labels)
        aspect_lens = np.asarray(aspect_lens)
        sentence_lens = np.asarray(sentence_lens)
        aspect_positions = np.asarray(aspect_positions)
        np.savez(save_fname, aspects=aspects, sentences=sentences, labels=labels,
                 aspect_lens=aspect_lens, sentence_lens=sentence_lens, aspect_positions=aspect_positions)
        return save_fname

def load_word_embeddings(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    word2vec = np.random.uniform(-0.01, 0.01, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2id:
                word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))
                oov = oov - 1
    word2vec[word2id['<pad>'], :] = 0
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec

