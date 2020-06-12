# Copyright 2020 The DDSP Authors.
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

# Lint as: python3
"""Tests for ddsp.losses."""

import pytest
from tddsp import spectral_ops
from tddsp.core import complex_abs
import numpy as np
import torch as th


def gen_np_sinusoid(frequency, amp, sample_rate, audio_len_sec):
    x = np.linspace(0, audio_len_sec, int(audio_len_sec * sample_rate))
    audio_sin = amp * (np.sin(2 * np.pi * frequency * x))
    return audio_sin


def gen_np_batched_sinusoids(
    frequency, amp, sample_rate, audio_len_sec, batch_size
):
    batch_sinusoids = [
        gen_np_sinusoid(frequency, amp, sample_rate, audio_len_sec)
        for _ in range(batch_size)
    ]
    return np.array(batch_sinusoids)


class TestSTFT:
    def test_th_and_np_are_consistent(self):
        amp = 1e-2
        audio = amp * (np.random.rand(64000).astype(np.float32) * 2.0 - 1.0)
        frame_size = 2048
        hop_size = 128
        overlap = 1.0 - float(hop_size) / frame_size
        pad_end = True

        s_np = spectral_ops.stft_np(
            audio, frame_size=frame_size, overlap=overlap, pad_end=pad_end
        )

        s_th = spectral_ops.stft(
            audio, frame_size=frame_size, overlap=overlap, pad_end=pad_end
        )

        # TODO(jesseengel): The phase comes out a little different, figure out why.
        assert np.allclose(
            np.abs(s_np), complex_abs(s_th), rtol=1e-3, atol=1e-3
        )


class TestDiff:
    def test_shape_is_correct(self):
        n_batch = 2
        n_time = 125
        n_freq = 100
        mag = th.ones([n_batch, n_freq, n_time])

        diff = spectral_ops.diff
        delta_t = diff(mag, axis=1)
        assert delta_t.shape[1] == mag.shape[1] - 1
        delta_delta_t = diff(delta_t, axis=1)
        assert delta_delta_t.shape[1] == mag.shape[1] - 2
        delta_f = diff(mag, axis=2)
        assert delta_f.shape[2] == mag.shape[2] - 1
        delta_delta_f = diff(delta_f, axis=2)
        assert delta_delta_f.shape[2] == mag.shape[2] - 2


class TestLoudness:
    def test_th_and_np_are_consistent(self):
        amp = 1e-2
        audio = amp * (np.random.rand(64000).astype(np.float32) * 2.0 - 1.0)
        frame_size = 2048
        frame_rate = 250

        ld_th = spectral_ops.compute_loudness(
            audio, n_fft=frame_size, frame_rate=frame_rate, use_th=True
        )

        ld_np = spectral_ops.compute_loudness(
            audio, n_fft=frame_size, frame_rate=frame_rate, use_th=False
        )

        assert np.allclose(np.abs(ld_np), ld_th.abs(), rtol=1e-3, atol=1e-3)


class TestPadOrTrimVectorToExpectedLength:
    @pytest.mark.parametrize(
        "use_th, num_dims",
        [(False, 1), (False, 2), (True, 1), (True, 2)],
        ids=["np_1d", "np_2d", "th_1d", "th_2d"],
    )
    def test_pad_or_trim_vector_to_expected_length(self, use_th, num_dims):
        vector_len = 10
        padded_vector_expected_len = 15
        trimmed_vector_expected_len = 4

        # Generate target vectors for testing
        vector = np.ones(vector_len) + np.random.uniform()
        num_pad = padded_vector_expected_len - vector_len
        target_padded = np.concatenate([vector, np.zeros(num_pad)])
        target_trimmed = vector[:trimmed_vector_expected_len]

        # Make a batch of target vectors
        if num_dims > 1:
            batch_size = 16
            vector = np.tile(vector, (batch_size, 1))
            target_padded = np.tile(target_padded, (batch_size, 1))
            target_trimmed = np.tile(target_trimmed, (batch_size, 1))

        vector_padded = spectral_ops.pad_or_trim_to_expected_length(
            vector, padded_vector_expected_len, use_th=use_th
        )
        vector_trimmmed = spectral_ops.pad_or_trim_to_expected_length(
            vector, trimmed_vector_expected_len, use_th=use_th
        )
        assert np.allclose(target_padded, vector_padded)
        assert np.allclose(target_trimmed, vector_trimmmed)


class TestComputeF0AndLoudness:
    def setup_method(self):
        """Creates some common default values for the test sinusoid."""
        self.amp = 0.75
        self.frequency = 440.0
        self.frame_rate = 250

    # @pytest.mark.parametrize(
    #     "sample_rate, audio_len_sec",
    #     [
    #         (16000, 0.21),
    #         (24000, 0.21),
    #         (44100, 0.21),
    #         (48000, 0.21),
    #         (16000, 0.4),
    #         (24000, 0.4),
    #         (44100, 0.4),
    #         (48000, 0.4),
    #     ],
    #     ids=[
    #         "16k_.21secs",
    #         "24k_.21secs",
    #         "44.1k_.21secs",
    #         "48k_.21secs",
    #         "16k_.4secs",
    #         "24k_.4secs",
    #         "44.1k_.4secs",
    #         "48k_.4secs",
    #     ],
    # )
    # def test_compute_f0_at_sample_rate(self, sample_rate, audio_len_sec):
    #     audio_sin = gen_np_sinusoid(
    #         self.frequency, self.amp, sample_rate, audio_len_sec
    #     )
    #     f0_hz, f0_confidence = spectral_ops.compute_f0(
    #         audio_sin, sample_rate, self.frame_rate
    #     )
    #     expected_f0_hz_and_f0_conf_len = int(self.frame_rate * audio_len_sec)
    #     assert len(f0_hz) == expected_f0_hz_and_f0_conf_len
    #     assert len(f0_confidence) == expected_f0_hz_and_f0_conf_len
    #     assert np.all(np.isfinite(f0_hz))
    #     assert np.all(np.isfinite(f0_confidence))

    @pytest.mark.parametrize(
        "sample_rate, audio_len_sec",
        [
            (16000, 0.21),
            (24000, 0.21),
            (48000, 0.21),
            (16000, 0.4),
            (24000, 0.4),
            (48000, 0.4),
        ],
        ids=[
            "16k_.21secs",
            "24k_.21secs",
            "48k_.21secs",
            "16k_.4secs",
            "24k_.4secs",
            "48k_.4secs",
        ],
    )
    def test_compute_loudness_at_sample_rate_1d(
        self, sample_rate, audio_len_sec
    ):
        audio_sin = gen_np_sinusoid(
            self.frequency, self.amp, sample_rate, audio_len_sec
        )
        expected_loudness_len = int(self.frame_rate * audio_len_sec)

        for use_th in [False, True]:
            loudness = spectral_ops.compute_loudness(
                audio_sin, sample_rate, self.frame_rate, use_th=use_th
            )
            if use_th:
                loudness = loudness.numpy()
            assert len(loudness) == expected_loudness_len
            assert np.all(np.isfinite(loudness))

    @pytest.mark.parametrize(
        "sample_rate, audio_len_sec",
        [
            (16000, 0.21),
            (24000, 0.21),
            (48000, 0.21),
            (16000, 0.4),
            (24000, 0.4),
            (48000, 0.4),
        ],
        ids=[
            "16k_.21secs",
            "24k_.21secs",
            "48k_.21secs",
            "16k_.4secs",
            "24k_.4secs",
            "48k_.4secs",
        ],
    )
    def test_compute_loudness_at_sample_rate_2d(
        self, sample_rate, audio_len_sec
    ):
        batch_size = 8
        audio_sin_batch = gen_np_batched_sinusoids(
            self.frequency, self.amp, sample_rate, audio_len_sec, batch_size
        )
        expected_loudness_len = int(self.frame_rate * audio_len_sec)

        for use_th in [False, True]:
            loudness_batch = spectral_ops.compute_loudness(
                audio_sin_batch, sample_rate, self.frame_rate, use_th=use_th
            )
            if use_th:
                loudness_batch = loudness_batch.numpy()

            assert loudness_batch.shape[0] == batch_size
            assert loudness_batch.shape[1] == expected_loudness_len
            assert np.all(np.isfinite(loudness_batch))

            # Check if batched loudness is equal to equivalent single computations
            audio_sin = gen_np_sinusoid(
                self.frequency, self.amp, sample_rate, audio_len_sec
            )
            loudness_target = spectral_ops.compute_loudness(
                audio_sin, sample_rate, self.frame_rate, use_th=use_th
            )
            loudness_batch_target = np.tile(loudness_target, (batch_size, 1))
            # Allow tolerance within 1dB
            assert np.allclose(
                loudness_batch, loudness_batch_target, atol=1, rtol=1
            )

    @pytest.mark.parametrize(
        "sample_rate, audio_len_sec",
        [
            (16000, 0.21),
            (24000, 0.21),
            (48000, 0.21),
            (16000, 0.4),
            (24000, 0.4),
            (48000, 0.4),
        ],
        ids=[
            "16k_.21secs",
            "24k_.21secs",
            "48k_.21secs",
            "16k_.4secs",
            "24k_.4secs",
            "48k_.4secs",
        ],
    )
    def test_th_compute_loudness_at_sample_rate(
        self, sample_rate, audio_len_sec
    ):
        audio_sin = gen_np_sinusoid(
            self.frequency, self.amp, sample_rate, audio_len_sec
        )
        loudness = spectral_ops.compute_loudness(
            audio_sin, sample_rate, self.frame_rate
        )
        expected_loudness_len = int(self.frame_rate * audio_len_sec)
        assert len(loudness) == expected_loudness_len
        assert np.all(np.isfinite(loudness))

    @pytest.mark.parametrize(
        "sample_rate, audio_len_sec",
        [(44100, 0.21), (44100, 0.4)],
        ids=["44.1k_.21secs", "44.1k_.4secs"],
    )
    def test_compute_loudness_indivisible_rates_raises_error(
        self, sample_rate, audio_len_sec
    ):
        audio_sin = gen_np_sinusoid(
            self.frequency, self.amp, sample_rate, audio_len_sec
        )

        for use_th in [False, True]:
            with pytest.raises(ValueError):
                spectral_ops.compute_loudness(
                    audio_sin, sample_rate, self.frame_rate, use_th=use_th
                )
