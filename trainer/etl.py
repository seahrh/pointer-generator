import os
import random

import en_core_web_sm
import stringx
import tensorflow as tf
import tensorflow.logging as log
from tensorflow.python.lib.io import file_io

nlp = en_core_web_sm.load()

# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]
STOPLIST = frozenset(['@highlight'])


def __int64_feature(value):
    value = value if type(value) == list else [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def __bytes_feature(value):
    value = value if type(value) == list else [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def __float_feature(value):
    value = value if type(value) == list else [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def __is_stopword(s):
    return s in STOPLIST


def contains_number(s):
    for c in s:
        if c.isdigit():
            return True
    return False


def __fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if '@highlight' in line:
        return line
    if line == '':
        return line
    for et in END_TOKENS:
        if line.endswith(et):
            return line
    return line + ' .'


def tokenize(s):
    res = []
    doc = nlp(s)
    for token in doc:
        t = token.text.strip().lower()
        if t == '' or __is_stopword(t):
            continue
        res.append(t)
    return res


def preprocess(s):
    s = stringx.to_str(s)
    sep = '\n'
    lines = s.split(sep)
    ls = []
    for line in lines:
        line = line.strip()
        line = stringx.to_ascii_str(line)
        # fix missing period must come after to_ascii conversion
        # because some punctuation falls outside ascii e.g. latex
        line = __fix_missing_period(line)
        ls.append(line)
    return tokenize(sep.join(ls))


def split_train_val_test(paths, train_size=0.7, test_size=0.1, shuffle=True):
    if shuffle:
        random.shuffle(paths)
    _len = len(paths)
    if train_size < 1:
        train_size = max(int(train_size * _len), 1)
    if test_size < 1:
        test_size = max(int(test_size * _len), 1)
    val_size = _len - train_size - test_size
    log.info('train_size={}, val_size={}, test_size={}'.format(repr(train_size), repr(val_size), repr(test_size)))
    train = set(paths[:train_size])
    val = set(paths[train_size:train_size + val_size])
    test = set(paths[-test_size:])
    intersect = train.intersection(val).intersection(test)
    if len(intersect) != 0:
        raise ValueError('intersect of train,val,test sets should be empty')
    return train, val, test


def article_example(article, abstract):
    article = stringx.to_bytes(article)
    abstract = stringx.to_bytes(abstract)
    return tf.train.Example(features=tf.train.Features(feature={
        'article': __bytes_feature(article),
        'abstract': __bytes_feature(abstract)
    }))


def __parse_proto(example_proto):
    features = {
        'article': tf.FixedLenFeature((), tf.string, default_value=''),
        'abstract': tf.FixedLenFeature((), tf.string, default_value='')
    }
    parsed = tf.parse_single_example(example_proto, features)
    return parsed['article'], parsed['abstract']


def __preprocess_article_and_abstract(article, abstract):
    sep = ' '
    return sep.join(preprocess(article)), sep.join(preprocess(abstract))


def dataset(data_path, batch_size=1, shuffle=False, repeat=False):
    names = file_io.list_directory(data_path)
    _paths = []
    for name in names:
        _paths.append(os.path.join(data_path, name))
    ds = tf.data.TFRecordDataset(_paths)
    ds = ds.map(__parse_proto)
    ds = ds.map(
        lambda article, abstract: tuple(tf.py_func(
            __preprocess_article_and_abstract,
            [article, abstract],
            [tf.string, tf.string],
            name='preprocess_article_and_abstract'
        )))
    if shuffle:
        ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size, drop_remainder=True)
    if repeat:
        ds = ds.repeat()
    return ds
