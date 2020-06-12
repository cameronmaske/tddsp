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
"""Tests for ddsp.effects."""

import pytest
from tddsp import effects
import torch as th


class TestReverb:
    def setup_method(self):
        """Creates some test specific attributes."""
        self.reverb_class = effects.Reverb
        self.audio = th.zeros((3, 16000))
        self.construct_args = {"reverb_length": 100}
        self.call_args = {"ir": th.zeros((3, 100, 1))}
        self.controls_keys = ["audio", "ir"]

    @pytest.mark.parametrize(
        "trainable", [True, False], ids=["trainable", "not_trainable"]
    )
    def test_output_shape_and_variables_are_correct(self, trainable):
        reverb = self.reverb_class(trainable=trainable, **self.construct_args)
        if trainable:
            output = reverb(self.audio)
        else:
            output = reverb(self.audio, **self.call_args)

        assert list(self.audio.shape) == list(output.shape)
        assert reverb.trainable == trainable
        params = list(reverb.parameters())
        nb_trainable_params = sum(
            p.numel() for p in params if p.requires_grad is True
        )
        nb_non_trainable_params = sum(
            p.numel() for p in params if p.requires_grad is False
        )
        assert nb_non_trainable_params == 0
        if trainable:
            assert nb_trainable_params != 0
        else:
            assert nb_trainable_params == 0

    def test_non_trainable_raises_value_error(self):
        reverb = self.reverb_class(trainable=False, **self.construct_args)
        with pytest.raises(ValueError):
            _ = reverb(self.audio)

    @pytest.mark.parametrize(
        "trainable", [True, False], ids=["trainable", "not_trainable"]
    )
    def test_get_controls_returns_correct_keys(self, trainable):
        reverb = self.reverb_class(trainable=trainable, **self.construct_args)
        if trainable:
            controls = reverb.get_controls(self.audio)
        else:
            controls = reverb.get_controls(self.audio, **self.call_args)

        assert list(controls.keys()) == self.controls_keys


class TestExpDecayReverb:
    def setup_method(self):
        """Creates some test specific attributes."""
        TestReverb.setup_method(self)
        self.reverb_class = effects.ExpDecayReverb
        self.audio = th.zeros((3, 16000))
        self.construct_args = {"reverb_length": 100}
        self.call_args = {"gain": th.zeros((3, 1)), "decay": th.zeros((3, 1))}


class TestFilteredNoiseReverb:
    def setup_method(self):
        """Creates some test specific attributes."""
        TestReverb.setup_method(self)
        self.reverb_class = effects.FilteredNoiseReverb
        self.audio = th.zeros((3, 16000))
        self.construct_args = {
            "reverb_length": 100,
            "n_frames": 10,
            "n_filter_banks": 20,
        }
        self.call_args = {"magnitudes": th.zeros((3, 10, 20))}


class TestFIRFilter:
    def test_output_shape_is_correct(self):
        processor = effects.FIRFilter()

        audio = th.zeros((3, 16000))
        magnitudes = th.zeros((3, 100, 30))
        output = processor(audio, magnitudes)

        assert [3, 16000] == list(output.shape)


class TestModDelay:
    @pytest.mark.parametrize(
        "channels", [True, False], ids=["with_channels", "without_channels"]
    )
    def test_output_shape_is_correct(self, channels):
        processor = effects.ModDelay()

        audio = th.zeros((3, 16000))
        gain = th.zeros((3, 16000))
        phase = th.zeros((3, 16000))
        if channels:
            gain.unsqueeze_(1)
            phase.unsqueeze_(1)
        output = processor(audio, gain, phase)

        assert [3, 16000] == list(output.shape)
