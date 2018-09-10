import argparse
import tensorflow as tf
import tensorflow.logging as log
import trainer.etl as etl


def log_batch(articles, abstracts):
    for i in range(len(articles)):
        article = articles[i]
        abstract = abstracts[i]
        log.info('i={}\n\narticle={}\n\nabstract={}'.format(i, repr(article), repr(abstract)))


def __main(data_path, max_step, batch_size, is_monitored_training_session):
    log.set_verbosity(log.INFO)
    ds = etl.dataset(data_path, batch_size)
    iterator = ds.make_one_shot_iterator()
    next_batch = iterator.get_next()
    if is_monitored_training_session:
        with tf.train.MonitoredTrainingSession() as sess:
            step = 0
            while not sess.should_stop() and step <= max_step:
                log.info('step={}'.format(step))
                articles, abstracts = sess.run(next_batch)
                log_batch(articles, abstracts)
                step += 1
        return
    with tf.Session() as sess:
        for step in range(max_step):
            log.info('step={}'.format(step))
            articles, abstracts = sess.run(next_batch)
            log_batch(articles, abstracts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Directory path containing .tfrecord files')
    parser.add_argument(
        '--max_step',
        type=int,
        default=20,
        help='model will be trained for maximum number of steps')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='minibatch size')
    parser.add_argument(
        '--is_monitored_training_session',
        type=bool,
        default=True,
        help='use monitored training session')
    args = parser.parse_args()
    __main(**vars(args))
