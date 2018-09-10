import argparse
import logging as log
import sys
import os
import time
import datetime
import collections
import tensorflow as tf
from trainer import etl

ENCODING = 'utf-8'

log.basicConfig(
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[log.StreamHandler(sys.stdout)],
    level=log.INFO
)


def __lines(text_file):
    res = []
    with open(text_file, 'r', encoding=ENCODING, errors='ignore') as f:
        for line in f:
            res.append(line.strip())
    return res


def __save_vocab_file(out_dir, vocab_counter, vocab_size):
    log.info('Saving vocab file...')
    _path = os.path.join(out_dir, 'vocab.tsv')
    with open(_path, 'w', encoding=ENCODING) as w:
        for word, count in vocab_counter.most_common(vocab_size):
            w.write(word + '\t' + str(count) + '\n')
    log.info('Saved vocab file [%s]', repr(_path))


def __article_abstract_tuple(file_path):
    lines = __lines(file_path)
    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        line = line.strip()
        if line == '':
            continue  # empty line
        if line.startswith("@highlight"):
            next_is_highlight = True
            continue
        if next_is_highlight:
            highlights.append(line)
            continue
        article_lines.append(line)
    return '\n'.join(article_lines), '\n'.join(highlights)


def __clean(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return
    names = os.listdir(directory)
    for name in names:
        os.remove(os.path.join(directory, name))


def __save_tfrecord(file_path, article, abstract):
    example = etl.article_example(article, abstract)
    log.debug('example={}'.format(repr(example)))
    with tf.python_io.TFRecordWriter(file_path) as writer:
        writer.write(example.SerializeToString())


def __save_article_and_abstract(file_path, article, abstract):
    with open(file_path, 'w', encoding=ENCODING) as writer:
        writer.write(article + '\n=====  ABSTRACT  =====\n' + abstract + '\n')
    log.info('Saved {}'.format(repr(file_path)))


def __out_filename(in_path, ext):
    """Get output filename from the input file path"""
    res = os.path.basename(in_path)
    res = os.path.splitext(res)[0]
    now = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
    res = res + '_' + now + '.' + ext
    return res


def __update_vocab(vocab_counter, terms):
    terms = [t for t in terms if not etl.contains_number(t)]
    vocab_counter.update(terms)


def __preprocess(paths, vocab_counter, batch_size, out_dir, save_article_and_abstract):
    count = 0
    t0 = time.time()
    _len = len(paths)
    for _path in paths:
        article, abstract = __article_abstract_tuple(_path)
        if save_article_and_abstract:
            out_path = os.path.join(out_dir, __out_filename(_path, ext='txt'))
            __save_article_and_abstract(out_path, article, abstract)
        out_path = os.path.join(out_dir, __out_filename(_path, ext='tfrecord'))
        __save_tfrecord(out_path, article, abstract)
        art_tokens = etl.preprocess(article)
        __update_vocab(vocab_counter, art_tokens)
        abs_tokens = etl.preprocess(abstract)
        __update_vocab(vocab_counter, abs_tokens)
        count += 1
        if count % batch_size == 0:
            t1 = time.time()
            log.info('story [%i of %i] - %.2f percent done (%i s)' % (
                count, _len, float(count) * 100.0 / float(_len), int(t1 - t0)))
            t0 = time.time()
    t1 = time.time()
    log.info('story [%i of %i] - %.2f percent done (%i s)' % (
        count, _len, float(count) * 100.0 / float(_len), int(t1 - t0)))


def __main(in_dirs, out_dir, vocab_size, batch_size, save_article_and_abstract):
    log.info('Args\nin_dirs={}\nout_dir={}'.format(repr(in_dirs), repr(out_dir)))
    train_dir = os.path.join(out_dir, 'train')
    __clean(train_dir)
    val_dir = os.path.join(out_dir, 'val')
    __clean(val_dir)
    test_dir = os.path.join(out_dir, 'test')
    __clean(test_dir)
    train = set()
    val = set()
    test = set()
    for in_dir in in_dirs:
        paths = []
        names = os.listdir(in_dir)
        for name in names:
            if name.startswith('.'):
                continue
            paths.append(os.path.join(in_dir, name))
        _train, _val, _test = etl.split_train_val_test(paths, train_size=0.95, test_size=0.01)
        train = train.union(_train)
        val = val.union(_val)
        test = test.union(_test)
    log.info('len(train)={}, len(val)={}, len(test)={}'.format(repr(len(train)), repr(len(val)), repr(len(test))))
    vocab_counter = collections.Counter()
    __preprocess(
        train,
        vocab_counter=vocab_counter,
        batch_size=batch_size,
        out_dir=train_dir,
        save_article_and_abstract=save_article_and_abstract
    )
    log.info('train set done')
    __preprocess(
        val,
        vocab_counter=vocab_counter,
        batch_size=batch_size,
        out_dir=val_dir,
        save_article_and_abstract=save_article_and_abstract
    )
    log.info('val set done')
    __preprocess(
        test,
        vocab_counter=vocab_counter,
        batch_size=batch_size,
        out_dir=test_dir,
        save_article_and_abstract=save_article_and_abstract
    )
    log.info('test set done')
    __save_vocab_file(out_dir=out_dir, vocab_counter=vocab_counter, vocab_size=vocab_size)
    log.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dirs',
        type=str,
        nargs='+',
        help='List of input directories',
        required=True
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        help='Output directory',
        required=True
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='batch size',
        default=100
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        help='Maximum number of terms in vocabulary',
        default=200000
    )
    parser.add_argument(
        '--save_article_and_abstract',
        type=bool,
        help='Save extracted article and abstract as .txt files',
        default=False
    )
    args = parser.parse_args()
    __main(**vars(args))
