#coding=utf-8
import re
from collections import  defaultdict
import csv
import numpy as np
import gensim
import pickle
import pandas as pd

# clean_data
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9,!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip().lower()

def build_data_cv(datafile, cv=10):
    revs = []
    word_freq = defaultdict(int)


    with open(datafile, "rb") as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        first_line = True
        for line in csvreader:
            if first_line:
             first_line = False
             continue
            status = []
            sentences = re.split(r'[.?]', line[1].strip()) #  add a spliting mark "!"
            try:
                sentences.remove('')
            except ValueError:
                None

            for sent in sentences:

                orig_rev = clean_str(sent.strip())
                if orig_rev == '':
                    continue
                words = set(orig_rev.split())
                splitted = orig_rev.split()
                if len(splitted) > 50:
                    orig_rev = []
                    splits = int(np.floor(len(splitted) / 20))
                    for index in range(splits):
                        orig_rev.append(' '.join(splitted[index*20: (index + 1) * 20]))
                    if len(splitted) > splits * 20:
                        orig_rev.append(' '.join(splitted[splits*20:]))
                    status.extend(orig_rev)
                    # print orig_rev
                else:
                    # None
                    # print orig_rev
                    status.append(orig_rev)
                # print orig_rev
                #     print len(orig_rev.split(' '))

                for word in words:
                    word_freq[word] += 1

            # print len(status)
            # total += len(status)
            # if len(status) > max_num:
            #     max_num = len(status)

            i = np.random.randint(0, cv)
            datum = {
                "y0" : 1 if line[2].lower() == 'y' else 0,
                "y1" : 1 if line[3].lower() == 'y' else 0,
                "y2" : 1 if line[4].lower() == 'y' else 0,
                "y3" : 1 if line[5].lower() == 'y' else 0,
                "y4" : 1 if line[6].lower() == 'y' else 0,
                "text" : status,
                "user" : line[0],
                "num_words": np.max([len(sent.split()) for sent in status]),
                "split": i
            }
            revs.append(datum)

        return revs, word_freq

# dataset, _ = build_data_cv('data/essays.csv', cv=10)

def get_vocab_and_W(word_freq, vocab_save_path=None, embedding_save_path=None):

    vocab = {}
    i = 1
    min_freq = 1
    vocab['UNK'] = 0
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = i
            i += 1
    print "vocab size:", len(vocab)

    if vocab_save_path is not None:
        with open(vocab_save_path, 'wb') as g:
            pickle.dump(vocab, g)

    embed_W = [np.random.uniform(-0.25, 0.25, 300) for j in range(len(vocab))]
    w2v_path = '/media/iiip/Elements/词向量语料/GoogleNews-vectors-negative300.bin'
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    num = 0
    for word in vocab:
        index = vocab[word]
        if word in w2v:
            print word
            num += 1
            embed_W[index] = np.array(w2v[word])
    embed_W = np.array(embed_W)
    print embed_W
    print 'vali word', num

    if embedding_save_path is not None:
        with open(embedding_save_path, 'wb') as g:
            pickle.dump(embed_W, g)


def make_idx_data_cv(revs, vocab_path, cv, maxnum, maxlen):
    vocab = pickle.load(open(vocab_path))
    features = get_mairesse()
    Y0_train, Y1_train, Y2_train, Y3_train, Y4_train= [], [], [], [], []
    Y0_test, Y1_test, Y2_test, Y3_test, Y4_test = [], [], [], [], []
    X_train, X_test = [], []
    X_ma_train, X_ma_test = [], []

    for sample in revs:
        x = np.zeros((maxnum, maxlen), dtype=np.int32)
        y0 = sample['y0']
        y1 = sample['y1']
        y2 = sample['y2']
        y3 = sample['y3']
        y4 = sample['y4']
        cv_id = sample['split']
        text = sample['text']
        user = sample['user']
        ma_features = features[user]
        for i in range(len(text)):
            if i < maxnum:
                sent = text[i]
                for j in range(len(sent)):
                    if j < maxlen:
                        word = sent[j]
                        if word in vocab:
                            index = vocab[word]
                        else:
                            index = 0
                        x[i][j] = index
        if cv_id == cv:
            Y0_test.append(y0)
            Y1_test.append(y1)
            Y2_test.append(y2)
            Y3_test.append(y3)
            Y4_test.append(y4)
            X_test.append(x)
            X_ma_test.append(ma_features)
        else:
            Y0_train.append(y0)
            Y1_train.append(y1)
            Y2_train.append(y2)
            Y3_train.append(y3)
            Y4_train.append(y4)
            X_train.append(x)
            X_ma_train.append(ma_features)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    # print Y0_train
    Y_train = [ Y0_train, Y1_train, Y2_train, Y3_train, Y4_train]
    Y_test = [ Y0_test, Y1_test, Y2_test, Y3_test, Y4_test ]
    return X_train, Y_train, X_test, Y_test, X_ma_train, X_ma_test


def get_mairesse():
    filename = 'data/mairesse.csv'
    feats = {}
    with open(filename, 'rb') as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        for line in csvreader:
            feats[line[0]] = ([float(f) for f in line[1:]])
    return feats

