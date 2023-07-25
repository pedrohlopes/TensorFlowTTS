# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team
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
"""Train Hifigan."""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

import sys

sys.path.append(".")

import argparse
import logging
import os

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

import tensorflow_tts
from examples.melgan.audio_log_mag_stft_dataset import AudioLogMagSTFTDataset
# from examples.melgan.train_melgan import collater # implementei a minha...
from examples.melgan_stft.train_melgan_stft import MultiSTFTMelganTrainer
from tensorflow_tts.configs import (
    HifiGANDiscriminatorConfig,
    HifiGANGeneratorConfig,
    MelGANDiscriminatorConfig,
)
from tensorflow_tts.models import (
    TFHifiGANGenerator,
    TFHifiGANMultiPeriodDiscriminator,
    TFMelGANMultiScaleDiscriminator,
)
from tensorflow_tts.utils import return_strategy


def collater(
    items,
    batch_max_steps=tf.constant(8192, dtype=tf.int32),
    hop_size=tf.constant(256, dtype=tf.int32),
):
    """Initialize collater (mapping function) for Tensorflow Audio-Mel Dataset.

    Args:
        batch_max_steps (int): The maximum length of input signal in batch.
        hop_size (int): Hop size of auxiliary features.

    """
    audio, log_mag_stft = items["audios"], items["log_mag_stfts"]

    if batch_max_steps is None:
        batch_max_steps = (tf.shape(audio)[0] // hop_size) * hop_size

    batch_max_frames = batch_max_steps // hop_size
    if len(audio) < len(log_mag_stft) * hop_size:
        audio = tf.pad(audio, [[0, len(log_mag_stft) * hop_size - len(audio)]])

    if len(log_mag_stft) > batch_max_frames:
        # randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = len(log_mag_stft) - batch_max_frames
        start_frame = tf.random.uniform(
            shape=[], minval=interval_start, maxval=interval_end, dtype=tf.int32
        )
        start_step = start_frame * hop_size
        audio = audio[start_step : start_step + batch_max_steps]
        log_mag_stft = log_mag_stft[start_frame : start_frame + batch_max_frames, :]
    else:
        audio = tf.pad(audio, [[0, batch_max_steps - len(audio)]])
        log_mag_stft = tf.pad(log_mag_stft, [[0, batch_max_frames - len(log_mag_stft)], [0, 0]])

    items = {
        "utt_ids": items["utt_ids"],
        "audios": audio,
        "log_mag_stfts": log_mag_stft,
        "log_mag_stft_lengths": len(log_mag_stft),
        "audio_lengths": len(audio),
    }

    return items


class TFHifiGANDiscriminator(tf.keras.Model):
    def __init__(self, multiperiod_dis, multiscale_dis, **kwargs):
        super().__init__(**kwargs)
        self.multiperiod_dis = multiperiod_dis
        self.multiscale_dis = multiscale_dis

    def call(self, x):
        outs = []
        period_outs = self.multiperiod_dis(x)
        scale_outs = self.multiscale_dis(x)
        outs.extend(period_outs)
        outs.extend(scale_outs)
        return outs


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Hifigan (See detail in examples/hifigan/train_hifigan.py)"
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        type=str,
        help="directory including training data. ",
    )
    parser.add_argument(
        "--dev-dir",
        default=None,
        type=str,
        help="directory including development data. ",
    )
    parser.add_argument(
        "--use-norm", default=1, type=int, help="use norm mels for training or raw."
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--generator_mixed_precision",
        default=0,
        type=int,
        help="using mixed precision for generator or not.",
    )
    parser.add_argument(
        "--discriminator_mixed_precision",
        default=0,
        type=int,
        help="using mixed precision for discriminator or not.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        nargs="?",
        help="path of .h5 melgan generator to load weights from",
    )
    args = parser.parse_args()

    # return strategy
    STRATEGY = return_strategy()

    # set mixed precision config
    if args.generator_mixed_precision == 1 or args.discriminator_mixed_precision == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    args.generator_mixed_precision = bool(args.generator_mixed_precision)
    args.discriminator_mixed_precision = bool(args.discriminator_mixed_precision)

    args.use_norm = bool(args.use_norm)

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # check arguments
    if args.train_dir is None:
        raise ValueError("Please specify --train-dir")
    if args.dev_dir is None:
        raise ValueError("Please specify either --valid-dir")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = tensorflow_tts.__version__
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:
        log_mag_stft_length_threshold = config["batch_max_steps"] // config[
            "hop_size"
        ] + 2 * config["hifigan_generator_params"].get("aux_context_window", 0)
    else:
        log_mag_stft_length_threshold = None

    if config["format"] == "npy":
        audio_query = "*-wave.npy"
        log_mag_stft_query = "*-log_mag_stft-feats.h5" if args.use_norm is False else "*-norm-feats.npy"
        audio_load_fn = np.load
        log_mag_stft_load_fn = np.load
    else:
        raise ValueError("Only npy are supported.")

    # define train/valid dataset
    train_dataset = AudioLogMagSTFTDataset(
        root_dir=args.train_dir,
        audio_query=audio_query,
        log_mag_stft_query=log_mag_stft_query,
        audio_load_fn=audio_load_fn,
        log_mag_stft_load_fn=log_mag_stft_load_fn,
        log_mag_stft_length_threshold=log_mag_stft_length_threshold,
    ).create(
        is_shuffle=config["is_shuffle"],
        map_fn=lambda items: collater(
            items,
            batch_max_steps=tf.constant(config["batch_max_steps"], dtype=tf.int32),
            hop_size=tf.constant(config["hop_size"], dtype=tf.int32),
        ),
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"]
        * STRATEGY.num_replicas_in_sync
        * config["gradient_accumulation_steps"],
    )

    valid_dataset = AudioLogMagSTFTDataset(
        root_dir=args.dev_dir,
        audio_query=audio_query,
        log_mag_stft_query=log_mag_stft_query,
        audio_load_fn=audio_load_fn,
        log_mag_stft_load_fn=log_mag_stft_load_fn,
        log_mag_stft_length_threshold=log_mag_stft_length_threshold,
    ).create(
        is_shuffle=config["is_shuffle"],
        map_fn=lambda items: collater(
            items,
            batch_max_steps=tf.constant(
                config["batch_max_steps_valid"], dtype=tf.int32
            ),
            hop_size=tf.constant(config["hop_size"], dtype=tf.int32),
        ),
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
    )

    # define trainer
    trainer = MultiSTFTMelganTrainer(
        steps=0,
        epochs=0,
        config=config,
        strategy=STRATEGY,
        is_generator_mixed_precision=args.generator_mixed_precision,
        is_discriminator_mixed_precision=args.discriminator_mixed_precision,
    )

    with STRATEGY.scope():
        # define generator and discriminator
        generator = TFHifiGANGenerator(
            HifiGANGeneratorConfig(**config["hifigan_generator_params"]),
            name="hifigan_generator",
        )

        multiperiod_discriminator = TFHifiGANMultiPeriodDiscriminator(
            HifiGANDiscriminatorConfig(**config["hifigan_discriminator_params"]),
            name="hifigan_multiperiod_discriminator",
        )
        multiscale_discriminator = TFMelGANMultiScaleDiscriminator(
            MelGANDiscriminatorConfig(
                **config["melgan_discriminator_params"],
                name="melgan_multiscale_discriminator",
            )
        )

        discriminator = TFHifiGANDiscriminator(
            multiperiod_discriminator,
            multiscale_discriminator,
            name="hifigan_discriminator",
        )

        # dummy input to build model.
        fake_log_mag_stfts = tf.random.uniform(shape=[1, 100, 513], dtype=tf.float32)
        y_hat = generator(fake_log_mag_stfts)
        discriminator(y_hat)

        if len(args.pretrained) > 1:
            generator.load_weights(args.pretrained)
            logging.info(
                f"Successfully loaded pretrained weight from {args.pretrained}."
            )

        generator.summary()
        discriminator.summary()

        # define optimizer
        generator_lr_fn = getattr(
            tf.keras.optimizers.schedules, config["generator_optimizer_params"]["lr_fn"]
        )(**config["generator_optimizer_params"]["lr_params"])
        discriminator_lr_fn = getattr(
            tf.keras.optimizers.schedules,
            config["discriminator_optimizer_params"]["lr_fn"],
        )(**config["discriminator_optimizer_params"]["lr_params"])

        gen_optimizer = tf.keras.optimizers.Adam(
            learning_rate=generator_lr_fn,
            amsgrad=config["generator_optimizer_params"]["amsgrad"],
        )
        dis_optimizer = tf.keras.optimizers.Adam(
            learning_rate=discriminator_lr_fn,
            amsgrad=config["discriminator_optimizer_params"]["amsgrad"],
        )

    trainer.compile(
        gen_model=generator,
        dis_model=discriminator,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
    )

    # start training
    try:
        trainer.fit(
            train_dataset,
            valid_dataset,
            saved_path=os.path.join(config["outdir"], "checkpoints/"),
            resume=args.resume,
        )
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
