# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""

import tensorflow as tf
from tensorflow import logging as log
import time
import os
import json


def load_ckpt(saver, sess, log_root, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir)
    and restore it to saver and sess, waiting 10 secs in the case of failure.
    Also returns checkpoint name.
    """
    latest_filename = "checkpoint_best" if ckpt_dir == "eval" else None
    ckpt_dir = os.path.join(log_root, ckpt_dir)
    ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
    log.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    #while True:
     #   try:
    saver.restore(sess, ckpt_state.model_checkpoint_path)
    return ckpt_state.model_checkpoint_path
      #  except:
       #     log.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
        #    time.sleep(10)


def __tf_config_json():
    conf = os.environ.get('TF_CONFIG')
    if not conf:
        return None
    return json.loads(conf)


def __session_config():
    """Returns a tf.ConfigProto instance that has appropriate device_filters set.
    """
    device_filters = None
    tf_config_json = __tf_config_json()
    if (tf_config_json and
            'task' in tf_config_json and
            'type' in tf_config_json['task'] and
            'index' in tf_config_json['task']):
        # Master should only communicate with itself and ps
        if tf_config_json['task']['type'] == 'master':
            device_filters = ['/job:ps', '/job:master']
        # Worker should only communicate with itself and ps
        elif tf_config_json['task']['type'] == 'worker':
            device_filters = [
                '/job:ps',
                '/job:worker/task:%d' % tf_config_json['task']['index']
            ]
    return tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=device_filters,
        operation_timeout_in_ms=120000,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )


def run_config(model_dir, random_seed):
    return tf.estimator.RunConfig(
        model_dir=model_dir,
        session_config=__session_config(),
        save_checkpoints_secs=60,
        save_summary_steps=100,
        keep_checkpoint_max=3,
        tf_random_seed=random_seed
    )


def repr_run_config(conf):
    assert isinstance(conf, tf.estimator.RunConfig)
    return """tf.estimator.RunConfig(
        model_dir={},
        cluster_spec={}, 
        is_chief={}, 
        master={}, 
        num_worker_replicas={}, 
        num_ps_replicas={}, 
        task_id={}, 
        task_type={},
        tf_random_seed={},
        save_summary_steps={},
        save_checkpoints_steps={},
        save_checkpoints_secs={},
        session_config={},
        keep_checkpoint_max={},
        keep_checkpoint_every_n_hours={},
        log_step_count_steps={},
        train_distribute={},
        device_fn={}
    )
    """.format(
        repr(conf.model_dir),
        repr(conf.cluster_spec.as_dict()),
        repr(conf.is_chief),
        repr(conf.master),
        repr(conf.num_worker_replicas),
        repr(conf.num_ps_replicas),
        repr(conf.task_id),
        repr(conf.task_type),
        repr(conf.tf_random_seed),
        repr(conf.save_summary_steps),
        repr(conf.save_checkpoints_steps),
        repr(conf.save_checkpoints_secs),
        repr(conf.session_config),
        repr(conf.keep_checkpoint_max),
        repr(conf.keep_checkpoint_every_n_hours),
        repr(conf.log_step_count_steps),
        repr(conf.train_distribute),
        repr(conf.device_fn)
    )
