# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
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
"""Dataset modules."""

import logging
import os

import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files


class LogMagSTFTDataset(AbstractDataset):


    def __init__(
        self,
        root_dir,
        log_mag_stft_query="*-log_mag_stft-feats.h5",
        log_mag_stft_load_fn=np.load,
        log_mag_stft_length_threshold=0,
    ):

        log_mag_stft_files = sorted(find_files(root_dir, log_mag_stft_query))
        log_mag_stft_lengths = [log_mag_stft_load_fn(f).shape[0] for f in log_mag_stft_files]

        assert len(log_mag_stft_files) != 0, f"Not found any log_mag_stft files in ${root_dir}."

        if ".npy" in log_mag_stft_query:
            suffix = log_mag_stft_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in log_mag_stft_files]

        # set global params
        self.utt_ids = utt_ids
        self.log_mag_stft_files = log_mag_stft_files
        self.log_mag_stft_lengths = log_mag_stft_lengths
        self.log_mag_stft_load_fn = log_mag_stft_load_fn
        self.log_mag_stft_length_threshold = log_mag_stft_length_threshold

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            log_mag_stft_file = self.log_mag_stft_files[i]
            log_mag_stft = self.log_mag_stft_load_fn(log_mag_stft_file)
            log_mag_stft_length = self.log_mag_stft_lengths[i]

            items = {"utt_ids": utt_id, "log_mag_stfts": log_mag_stft, "log_mag_stft_lengths": log_mag_stft_length}

            yield items

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "log_mag_stfts": tf.float32,
            "log_mag_stft_lengths": tf.int32,
        }
        return output_types

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )

        datasets = datasets.filter(
            lambda x: x["log_mag_stft_lengths"] > self.log_mag_stft_length_threshold
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padded shapes
        padded_shapes = {
            "utt_ids": [],
            "log_mag_stfts": [None, 80],
            "log_mag_stft_lengths": [],
        }

        datasets = datasets.padded_batch(batch_size, padded_shapes=padded_shapes)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "LogMagSTFTDataset"
