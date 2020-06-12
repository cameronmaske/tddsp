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
"""Library of functions for differentiable digital signal processing (DDSP)."""

import collections
from typing import Any, Dict, Text, TypeVar

import torch as th
import numpy as np
from scipy import fftpack

Number = TypeVar("Number", int, float, np.ndarray, th.Tensor)


def f32(*inputs):
    outputs = []
    for x in inputs:
        if th.is_tensor(x):
            if x.dtype != th.float:
                x = x.float()
        else:
            x = th.tensor(x, dtype=th.float)
        outputs.append(x)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def make_iterable(x):
    if x is None:
        return []
    else:
        return x if isinstance(x, collections.Iterable) else [x]


def nested_lookup(
    nested_key: Text, nested_dict: Dict[Text, Any], delimiter: Text = "/"
) -> th.Tensor:
    """Returns the value of a nested dict according to a parsed input string.

    Args:
        nested_key: String of the form "key/key/key...".
        nested_dict: Nested dictionary.
        delimiter: String that splits the nested keys.

    Returns:
        value: Value of the key from the nested dictionary.
    """
    # Parse the input string.
    keys = nested_key.split(delimiter)
    # Return the nested value.
    value = nested_dict
    for key in keys:
        value = value[key]
    return value


def midi_to_hz(notes: Number) -> Number:
    notes = f32(notes)
    return 440 * (2 ** ((notes - 69) / 12))


def hz_to_midi(frequencies: Number) -> Number:
    frequencies, A = f32(frequencies, 440)
    notes = 12 * (frequencies.log2() - A.log2()) + 69
    notes[notes == -np.inf] = 0
    return notes


def unit_to_midi(
    unit: Number,
    midi_min: Number = 20.0,
    midi_max: Number = 90.0,
    clip: bool = False,
) -> Number:
    unit = f32(unit)
    if clip:
        unit = unit.clamp(min=0, max=1)
    return midi_min + (midi_max - midi_min) * unit


def midi_to_unit(
    midi: Number,
    midi_min: Number = 20.0,
    midi_max: Number = 90.0,
    clip: bool = False,
) -> Number:
    unit = (midi - midi_min) / (midi_max - midi_min)
    unit = f32(unit)
    if clip:
        unit = unit.clamp(min=0, max=1)
    return unit


def unit_to_hz(
    unit: Number, hz_min: Number, hz_max: Number, clip: bool = False
) -> Number:
    midi = unit_to_midi(
        unit,
        midi_min=hz_to_midi(hz_min),
        midi_max=hz_to_midi(hz_max),
        clip=clip,
    )
    return midi_to_hz(midi)


def hz_to_unit(
    hz: Number, hz_min: Number, hz_max: Number, clip: bool = False
) -> Number:
    midi = hz_to_midi(hz)
    return midi_to_unit(
        midi,
        midi_min=hz_to_midi(hz_min),
        midi_max=hz_to_midi(hz_max),
        clip=clip,
    )


def resample(
    inputs: th.Tensor,
    n_timesteps: int,
    method: Text = "linear",
    add_endpoint: bool = True,
) -> th.Tensor:
    """Interpolates a tensor from n_frames to n_timesteps.

    Args:
        inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
            [batch_size, n_frames], [batch_size, channels, n_frames], or
            [batch_size, channels, n_freq, n_frames].
        n_timesteps: Time resolution of the output signal.
        method: Type of resampling, must be in ['nearest', 'linear', 'cubic',
            'window'].
            Linear and cubic ar typical bilinear, bicubic interpolation.
            'window' uses overlapping windows (only for upsampling) which is
            smoother for amplitude envelopes with large frame sizes.
        add_endpoint: Hold the last timestep for an additional step as the
            endpoint. Then, n_timesteps is divided evenly into n_frames
            segments. If false, use the last timestep as the endpoint,
            producing (n_frames - 1) segments with each having a length of
            n_timesteps / (n_frames - 1).

    Returns:
        Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
            [batch_size, n_timesteps], [batch_size, channels, n_timesteps],
            or [batch_size, channels, n_freqs, n_timesteps].

    Raises:
        ValueError: If method is 'window' and input is 4-D.
        ValueError: If method is not one of 'nearest', 'bilinear', 'bicubic',
            or 'window'.
    """
    inputs = f32(inputs)

    n_original = inputs.shape[-1]

    if n_original == n_timesteps:
        return inputs

    is_1d = inputs.dim() == 1  # [N]
    is_2d = inputs.dim() == 2  # [B x N]
    is_4d = inputs.dim() == 4  # [B x C x F x N]

    if is_1d:
        inputs = inputs[None, None, :]
    elif is_2d:
        inputs = inputs[:, None, :]

    def interpolate(method):
        outputs = inputs[:, :, None, :] if not is_4d else inputs
        size = th.Size([outputs.shape[2], n_timesteps])
        align_corners = None if method == "nearest" else not add_endpoint
        outputs = th.nn.functional.interpolate(
            outputs, size=size, mode=method, align_corners=align_corners
        )
        # # mimic tensorflow's align_corners behaviour
        # if add_endpoint and tf_behaviour:
        #     n_total = int(n_timesteps / n_original * (n_original - 1))
        #     n_shift = (n_timesteps - n_total) // 2
        #     outputs[..., :n_shift] = outputs[..., -n_shift:]
        #     outputs = outputs.roll(-n_shift, -1)
        return outputs[:, :, 0, :] if not is_4d else outputs

    if method == "nearest":
        outputs = interpolate("nearest")
    elif method == "linear":
        outputs = interpolate("bilinear")
    elif method == "cubic":
        outputs = interpolate("bicubic")
    elif method == "window":
        outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
    else:
        raise ValueError(
            f"Method ({method}) is invalid. Must be one of "
            "['nearest', 'linear', 'cubic', 'window']"
        )

    if is_1d:
        outputs = outputs[0, 0, :]
    elif is_2d:
        outputs = outputs[:, 0, :]

    return outputs


def upsample_with_windows(
    inputs: th.Tensor, n_timesteps: int, add_endpoint: bool = True
) -> th.Tensor:
    """Upsample a series of frames using using overlapping hann windows.
        Good for amplitude envelopes.

    Args:
        inputs: Framewise 3-D tensor. Shape [batch_size, n_channels, n_frames].
        n_timesteps: The time resolution of the output signal.
        add_endpoint: Hold the last timestep for an additional step as the
            endpoint. Then, n_timesteps is divided evenly into n_frames
            segments. If false, use the last timestep as the endpoint,
            producing (n_frames - 1) segments with each having a length of
            n_timesteps / (n_frames - 1).

    Returns:
        Upsampled 3-D tensor. Shape [batch_size, n_channels, n_timesteps].

    Raises:
        ValueError: If input does not have 3 dimensions.
        ValueError: If attempting to use function for downsampling.
        ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint
            is true) or n_frames - 1 (if add_endpoint is false).
    """
    inputs = f32(inputs)

    if inputs.dim() != 3:
        raise ValueError(
            "upsample_with_windows() only supports 3 dimensions, "
            f"not {inputs.shape}"
        )

    n_frames = inputs.shape[-1]
    n_intervals = n_frames - int(not add_endpoint)

    if n_frames >= n_timesteps:
        raise ValueError(
            "upsample_with_windows() cannot be used for downsampling"
        )

    if n_timesteps % n_intervals != 0:
        minus_one = "" if add_endpoint else " - 1"
        raise ValueError(
            "For upsampling, the target number of timesteps must be divisible "
            f"by the number of inputs frames{minus_one}."
        )

    if not add_endpoint:
        original_hop_size = n_timesteps // n_intervals
        n_timesteps += original_hop_size

    hop_size = n_timesteps // n_frames
    window_length = 2 * hop_size
    window = th.hann_window(window_length)[:, None]

    x = inputs[:, :, :, None]
    n_channels = x.shape[1]
    windowed = x.view(-1, 1, x.shape[-2]) * window
    out_size = (n_frames - 1) * hop_size + window_length
    outputs = th.nn.functional.fold(
        windowed, (1, out_size), (1, window_length), stride=(1, hop_size)
    )
    ones = th.ones_like(windowed) * window
    divisor = th.nn.functional.fold(
        ones, (1, out_size), (1, window_length), stride=(1, hop_size)
    )
    outputs = outputs / divisor
    if add_endpoint:
        hop_slice = slice(hop_size // 2, -hop_size // 2, None)
    else:
        hop_slice = slice(
            (hop_size + original_hop_size) // 2,
            -(hop_size + original_hop_size) // 2,
            None,
        )
        n_timesteps -= original_hop_size
    return outputs[:, :, 0, hop_slice].view(-1, n_channels, n_timesteps)


def log_scale(x: th.Tensor, min_x, max_x) -> th.Tensor:
    """Scales a -1 to 1 value logarithmically between min_x and max_x."""
    x, min_x, max_x = f32(x, min_x, max_x)
    x = (x + 1) / 2  # [-1, 1] -> [0, 1]
    return ((1 - x) * min_x.log() + x * max_x.log()).exp()


def exp_sigmoid(
    x: th.Tensor, exponent=10.0, max_value=2.0, threshold=1e-7
) -> th.Tensor:
    x, exponent = f32(x, exponent)
    return max_value * th.sigmoid(x).pow(exponent.log()) + threshold


def sym_exp_sigmoid(x: th.Tensor, width=8) -> th.Tensor:
    x = f32(x)
    return exp_sigmoid(width * (x.abs() / 2 - 1))


def remove_above_nyquist(
    frequency_envelopes: th.Tensor,
    amplitude_envelopes: th.Tensor,
    sample_rate: int = 44_100,
) -> th.Tensor:
    frequency_envelopes, amplitude_envelopes = f32(
        frequency_envelopes, amplitude_envelopes
    )

    above_nyquist = frequency_envelopes >= sample_rate / 2
    amplitude_envelopes[above_nyquist.expand_as(amplitude_envelopes)] = 0
    return amplitude_envelopes


def oscillator_bank(
    frequency_envelopes: th.Tensor,
    amplitude_envelopes: th.Tensor,
    sample_rate: int = 44_100,
    sum_sinusoids: bool = True,
) -> th.Tensor:
    """Generates audio from sample-wise frequencies for a bank of oscillators.

    Args:
        frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
            [batch_size, n_sinusoids, n_samples].
        amplitude_envelopes: Sample-wise oscillator amplitude. Shape
            [batch_size, n_sinusoids, n_samples].
        sample_rate: Sample rate in samples per a second.
        sum_sinusoids: Add up audio from all the sinusoids.

    Returns:
        wav: Sample-wise audio. Shape [batch_size, n_sinusoids, n_samples] if
            sum_sinusoids=False, else shape is [batch_size, n_samples].
    """
    frequency_envelopes, amplitude_envelopes = f32(
        frequency_envelopes, amplitude_envelopes
    )

    amplitude_envelopes = remove_above_nyquist(
        frequency_envelopes, amplitude_envelopes, sample_rate
    )
    omegas = frequency_envelopes * (2.0 * np.pi) / sample_rate
    phases = omegas.cumsum(axis=2)
    wavs = phases.sin()
    audio = amplitude_envelopes * wavs
    if sum_sinusoids:
        audio = audio.sum(-2)
    return audio


def get_harmonic_frequencies(
    frequencies: th.Tensor, n_harmonics: int
) -> th.Tensor:
    frequencies = f32(frequencies)
    f_ratios = th.linspace(1.0, n_harmonics, n_harmonics)[None, :, None]
    return frequencies * f_ratios


def harmonic_synthesis(
    frequencies: th.Tensor,
    amplitudes: th.Tensor,
    harmonic_shifts: th.Tensor = None,
    harmonic_distribution: th.Tensor = None,
    n_samples: int = 64_000,
    sample_rate: int = 44_100,
    amp_resample_method: Text = "window",
) -> th.Tensor:
    """Generate audio from frame-wise monophonic harmonic oscillator bank.

    Args:
        frequencies: Frame-wise fundamental frequency in Hz. Shape [batch_size,
            1, n_frames].
        amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size,
            1, n_frames].
        harmonic_shifts: Harmonic frequency variations (Hz), zero-centered.
            Total frequency of a harmonic is equal to
            (frequencies * harmonic_number * (1 + harmonic_shifts)).
            Shape [batch_size, n_harmonics, n_frames].
        harmonic_distribution: Harmonic amplitude variations, ranged zero to
            one. Total amplitude of a harmonic is equal to
            (amplitudes * harmonic_distribution).
            Shape [batch_size, n_harmonics, n_frames].
        n_samples: Total length of output audio. Interpolates and crops to this.
        sample_rate: Sample rate.
        amp_resample_method: Mode with which to resample amplitude envelopes.

    Returns:
        audio: Output audio. Shape [batch_size, 1, n_samples]
    """
    frequencies, amplitudes = f32(frequencies, amplitudes)

    if harmonic_distribution is not None:
        harmonic_distribution = f32(harmonic_distribution)
        n_harmonics = harmonic_distribution.shape[-2]
    elif harmonic_shifts is not None:
        harmonic_shifts = f32(harmonic_shifts)
        n_harmonics = harmonic_shifts.shape[-2]
    else:
        n_harmonics = 1

    harmonic_frequencies = get_harmonic_frequencies(frequencies, n_harmonics)
    if harmonic_shifts is not None:
        harmonic_frequencies *= 1 + harmonic_shifts

    if harmonic_distribution is not None:
        harmonic_amplitudes = amplitudes * harmonic_distribution
    else:
        harmonic_amplitudes = amplitudes

    frequency_envelopes = resample(harmonic_frequencies, n_samples)
    amplitude_envelopes = resample(
        harmonic_amplitudes, n_samples, method=amp_resample_method
    )

    return oscillator_bank(
        frequency_envelopes, amplitude_envelopes, sample_rate=sample_rate
    )


# Wavetable Synthesizer --------------------------------------------------------
def linear_lookup(phase: th.Tensor, wavetables: th.Tensor) -> th.Tensor:
    """Lookup from wavetables with linear interpolation.

    Args:
        phase: The instantaneous phase of the base oscillator, ranging from 0 to
            1.0. This gives the position to lookup in the wavetable.
            Shape [batch_size, 1, n_samples] or [batch_size, n_samples].
        wavetables: Wavetables to be read from on lookup. Shape [batch_size,
            n_wavetable, n_samples] or [batch_size, n_wavetable].

    Returns:
        The resulting audio from linearly interpolated lookup of the wavetables
            at each point in time. Shape [batch_size, n_samples].
    """
    phase, wavetables = f32(phase, wavetables)

    if wavetables.dim() == 2:
        wavetables = wavetables[:, :, None]

    if phase.dim() == 2:
        phase = phase[:, None, :]

    wavetables = th.cat([wavetables, wavetables[:, :1]], dim=-2)
    n_wavetable = wavetables.shape[-2]
    phase_wavetables = th.linspace(0, 1, n_wavetable)
    phase_distance = (phase - phase_wavetables[None, :, None]).abs()
    phase_distance *= n_wavetable - 1
    weights = th.relu(1 - phase_distance)
    weighted_wavetables = weights * wavetables
    return weighted_wavetables.sum(-2)


def wavetable_synthesis(
    frequencies: th.Tensor,
    amplitudes: th.Tensor,
    wavetables: th.Tensor,
    n_samples: int = 64000,
    sample_rate: int = 16000,
):
    """Monophonic wavetable synthesizer.

    Args:
        frequencies: Frame-wise frequency in Hertz of the fundamental
            oscillator. Shape [batch_size, n_frames] or
            [batch_size, 1, n_frames].
        amplitudes: Frame-wise amplitude envelope to apply to the oscillator.
            Shape [batch_size, n_frames] or [batch_size, 1, n_frames].
        wavetables: Frame-wise wavetables from which to lookup.
            Shape [batch_size, n_wavetable] or
            [batch_size, n_wavetable, n_frames].
        n_samples: Total length of output audio. Interpolates and crops to this.
        sample_rate: Number of samples per a second.

    Returns:
        audio: Audio at the frequency and amplitude of the inputs, with
            harmonics given by the wavetable. Shape [batch_size, n_samples].
    """
    wavetables, frequencies, amplitudes = f32(
        wavetables, frequencies, amplitudes
    )

    if frequencies.dim() == 3:
        frequencies = frequencies[:, 0, :]

    if amplitudes.dim() == 3:
        amplitudes = amplitudes[:, 0, :]

    if wavetables.dim() == 2:
        wavetables = wavetables[:, :, None]

    frequency_envelope = resample(frequencies, n_samples)
    amplitude_envelope = resample(amplitudes, n_samples, method="window")
    if wavetables.shape[2] > 1:
        wavetables = resample(wavetables, n_samples)

    phase_velocity = frequency_envelope / sample_rate
    phase = phase_velocity.cumsum(1).remainder(1.0)
    audio = linear_lookup(phase, wavetables)
    return audio * amplitude_envelope


def variable_length_delay(
    phase: th.Tensor, audio: th.Tensor, max_length: int = 512
) -> th.Tensor:
    """Delay audio by a time-vaying amount using linear interpolation.
    Useful for modulation effects such as vibrato, chorus, and flanging.

    Args:
        phase: The normalzed instantaneous length of the delay, ranging from 0
            to 1.0. This corresponds to a delay of 0 to max_length samples.
            Shape [batch_size, 1, n_samples].
        audio: Audio signal to be delayed. Shape [batch_size, n_samples].
        max_length: Maximimum delay in samples.

    Returns:
        The delayed audio signal. Shape [batch_size, n_samples].
    """
    phase, audio = f32(phase, audio)

    # Make causal by zero-padding audio up front.
    audio = th.nn.functional.pad(audio, (max_length - 1, 0))

    # Cut audio up into frames of max_length.
    frames = audio.unfold(-1, max_length, 1)
    # Reverse frames so that [0, 1] phase corresponds to [0, max_length] delay.
    frames = frames.flip(-1)
    # Reshape to [batch_size, max_length, n_frames]
    frames = frames.transpose(1, 2)
    return linear_lookup(phase, frames)


# Time-varying convolution -----------------------------------------------------
def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    """Calculate final size for efficient FFT.

    Args:
        frame_size: Size of the audio frame.
        ir_size: Size of the convolving impulse response.
        power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
            numbers. TPU requires power of 2, while GPU is more flexible.

    Returns:
        fft_size: Size for efficient FFT.
    """
    convolved_frame_size = f32(ir_size + frame_size - 1).squeeze()
    if power_of_2:
        fft_size = (2 ** convolved_frame_size.log2().ceil()).int().item()
    else:
        fft_size = fftpack.helper.next_fast_len(convolved_frame_size.int())
    return fft_size


def crop_and_compensate_delay(
    audio: th.Tensor,
    audio_size: int,
    ir_size: int,
    padding: Text,
    delay_compensation: int,
) -> th.Tensor:
    """Crop audio output from convolution to compensate for group delay.

    Args:
        audio: Audio after convolution. Tensor of shape [batch_size, time_steps].
        audio_size: Initial size of the audio before convolution.
        ir_size: Size of the convolving impulse response.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the
            audio is extended to include the tail of the impulse response
            (audio_timesteps + ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to
            compensate for group delay of the impulse response.
            If delay_compensation < 0 it defaults to automatically calculating
            a constant group delay of the windowed linear phase filter from
            frequency_impulse_response().

    Returns:
        Tensor of cropped and shifted audio.

    Raises:
        ValueError: If padding is not either 'valid' or 'same'.
    """
    if padding == "valid":
        crop_size = ir_size + audio_size - 1
    elif padding == "same":
        crop_size = audio_size
    else:
        raise ValueError(
            f"Padding must be 'valid' or 'same', instead of {padding}"
        )

    total_size = audio.shape[-1]
    crop = total_size - crop_size
    start = (
        (ir_size - 1) // 2 - 1 if delay_compensation < 0 else delay_compensation
    )
    end = crop - start
    return audio[:, start:-end]


def complex_multiplication(t1, t2):
    real1, imag1 = t1.split(1, -1)
    real2, imag2 = t2.split(1, -1)
    return th.cat(
        [real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1
    )


def complex_abs(t):
    real, imag = t.split(1, -1)
    return (real ** 2 + imag ** 2).sqrt()[..., 0]


def fft_convolve(
    audio: th.Tensor,
    impulse_response: th.Tensor,
    padding: Text = "same",
    delay_compensation: int = -1,
) -> th.Tensor:
    """Filter audio with frames of time-varying impulse responses.
    Time-varying filter. Given audio [batch_size, n_samples], and a series of
    impulse responses [batch_size, n_frames, n_impulse_response], splits the
    audio into frames, applies filters, and then overlap-and-adds audio back
    together. Applies non-windowed non-overlapping STFT/ISTFT to efficiently
    compute convolution for large impulse response sizes.

    Args:
        audio: Input audio. Tensor of shape [batch_size, audio_timesteps].
        impulse_response: Finite impulse response to convolve. Can either be a
            2-D Tensor of shape [batch_size, ir_size], or a 3-D Tensor of shape
            [batch_size, ir_frames, ir_size]. A 2-D tensor will apply a single
            linear time-invariant filter to the audio. A 3-D Tensor will apply a
            linear time-varying filter. Automatically chops the audio into equally
            shaped blocks to match ir_frames.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the
            audio is extended to include the tail of the impulse response
            (audio_timesteps + ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to
            compensate for group delay of the impulse response.
            If delay_compensation is less than 0 it defaults to automatically
            calculating a constant group delay of the windowed linear phase
            filter from frequency_impulse_response().

    Returns:
        audio_out: Convolved audio.
            Shape [batch_size, audio_timesteps + ir_timesteps - 1] ('valid'
            padding) or shape [batch_size, audio_timesteps] ('same' padding).

    Raises:
        ValueError: If audio and impulse response have different batch size.
        ValueError: If audio cannot be split into evenly spaced frames. (i.e.
            the number of impulse response frames is on the order of the audio
            size and not a multiple of the audio size.)
    """
    audio, impulse_response = f32(audio, impulse_response)

    if impulse_response.dim() == 2:
        impulse_response = impulse_response[:, None, :]

    batch_size_ir, n_ir_frames, ir_size = impulse_response.shape
    batch_size, audio_size = audio.shape

    if batch_size != batch_size_ir:
        raise ValueError(
            f"Batch size of audio ({batch_size}) and impulse response "
            f"({batch_size_ir}) must be the same."
        )

    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size
    padded_audio = th.nn.functional.pad(audio, (0, audio_size % hop_size))
    audio_frames = padded_audio.unfold(-1, frame_size, hop_size)

    n_audio_frames = audio_frames.shape[1]
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            f"Number of audio frames ({n_audio_frames}) and impulse response "
            f"frames ({n_ir_frames}) do not match. For small hop size = ceil("
            "audio_size / n_ir_frames), number of impulse response frames must "
            "be a multiple of the audio size."
        )

    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    audio_frames_size = audio_frames.shape[-1]
    if audio_frames_size < fft_size:
        audio_frames = th.nn.functional.pad(
            audio_frames, (0, fft_size - audio_frames_size)
        )
    elif audio_frames_size > fft_size:
        audio_frames = audio_frames[..., :fft_size]

    if ir_size < fft_size:
        impulse_response = th.nn.functional.pad(
            impulse_response, (0, fft_size - ir_size)
        )
    elif ir_size > fft_size:
        impulse_response = impulse_response[:, :fft_size, :]

    audio_fft = audio_frames.rfft(1)
    ir_fft = impulse_response.rfft(1)
    audio_ir_fft = complex_multiplication(audio_fft, ir_fft)
    th_fft_size = th.Size([fft_size])
    audio_frames_out = audio_ir_fft.irfft(1, signal_sizes=th_fft_size)
    audio_out_size = (n_audio_frames - 1) * hop_size + fft_size
    if audio_out_size > fft_size:
        audio_frames_out = th.nn.functional.pad(
            audio_frames_out,
            (0, audio_out_size - fft_size - n_audio_frames + 1),
        )
    audio_out = th.nn.functional.fold(
        audio_frames_out, (1, audio_out_size), (1, n_ir_frames)
    )
    ones = th.ones_like(audio_frames_out)
    divisor = th.nn.functional.fold(ones, (1, audio_out_size), (1, n_ir_frames))
    audio_out = audio_out / divisor
    audio_out = audio_out[:, 0, 0, :]
    return crop_and_compensate_delay(
        audio_out, audio_size, ir_size, padding, delay_compensation
    )


# Filter Design ----------------------------------------------------------------
def fftshift(x: th.Tensor, axis=-1) -> th.Tensor:
    return x.roll(x.shape[axis] // 2, axis)


def apply_window_to_impulse_response(
    impulse_response: th.Tensor, window_size: int = 0, causal: bool = False
) -> th.Tensor:
    """Apply a window to an impulse response and put in causal form.

    Args:
        impulse_response: A series of impulse responses frames to window, of
            shape [batch_size, n_frames, ir_size].
        window_size: Size of the window to apply in the time domain. If
            window_size is less than 1, it defaults to the impulse_response
            size.
        causal: Impulse responnse input is in causal form (peak in the middle).

    Returns:
        impulse_response: Windowed impulse response in causal form, with last
            dimension cropped to window_size if window_size is greater than 0
            and less than ir_size.
    """
    impulse_response = f32(impulse_response)
    if causal:
        impulse_response = fftshift(impulse_response)

    ir_size = impulse_response.shape[-1]
    if window_size <= 0 or window_size > ir_size:
        window_size = ir_size
    window = th.hann_window(window_size)

    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = th.cat(
            [window[half_idx:], th.zeros(padding), window[:half_idx]], dim=-1
        )
    else:
        window = fftshift(window)

    window = window.expand_as(impulse_response)
    impulse_response = window * th.real(impulse_response)

    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = th.cat(
            [
                impulse_response[..., first_half_start:],
                impulse_response[..., :second_half_end],
            ],
            dim=-1,
        )
    else:
        impulse_response = fftshift(impulse_response)
    return impulse_response


def frequency_impulse_response(
    magnitudes: th.Tensor, window_size: int = 0
) -> th.Tensor:
    """Get windowed impulse responses using the frequency sampling method.
    Follows the approach in:
    https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

    Args:
        magnitudes: Frequency transfer curve. Float32 Tensor of shape
            [batch_size, n_frames, n_frequencies] or [batch_size, n_frequencies].
            The frequencies of the last dimension are ordered as
            [0, f_nyqist / (n_frames -1), ..., f_nyquist], where f_nyquist is
            (sample_rate / 2). Automatically splits the audio into equally
            sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If
            window_size is less than 1, it defaults to the impulse_response
            size.

    Returns:
        impulse_response: Time-domain FIR filter of shape
            [batch_size, frames, window_size] or [batch_size, window_size].

    Raises:
        ValueError: If window size is larger than fft size.
    """
    magnitudes = f32(magnitudes)
    output_size = th.Size([(magnitudes.shape[-1] - 1) * 2])
    magnitudes = th.stack([magnitudes, th.zeros_like(magnitudes)], -1)
    impulse_response = th.irfft(magnitudes, 1, signal_sizes=output_size)
    return apply_window_to_impulse_response(impulse_response, window_size)


def sinc(x, threshold=1e-20):
    """Normalized zero phase version (peak at zero)."""
    x = f32(x)
    x[x.abs() < threshold] = threshold
    x *= np.pi
    return x.sin() / x


def sinc_impulse_response(cutoff_frequency, window_size=512, sample_rate=None):
    """Get a sinc impulse response for a set of low-pass cutoff frequencies.

    Args:
        cutoff_frequency: Frequency cutoff for low-pass sinc filter. If the
            sample_rate is given, cutoff_frequency is in Hertz. If sample_rate
            is None, cutoff_frequency is normalized ratio (frequency/nyquist)
            in the range [0, 1.0]. Shape [batch_size, n_time, 1].
        window_size: Size of the Hamming window to apply to the impulse.
        sample_rate: Optionally provide the sample rate.

    Returns:
        impulse_response: A series of impulse responses. Shape
            [batch_size, n_time, (window_size // 2) * 2 + 1].
    """
    if sample_rate is not None:
        cutoff_frequency *= 2 / sample_rate
    half_size = window_size // 2
    full_size = half_size * 2 + 1
    idx = th.arange(-half_size, half_size + 1, dtype=th.float)[None, None, :]

    impulse_response = sinc(cutoff_frequency * idx)
    window = th.hamming_window(full_size).expand_as(impulse_response)
    impulse_response = window * th.real(impulse_response)
    return impulse_response / impulse_response.sum(-1, keepdim=True)


def frequency_filter(
    audio: th.Tensor,
    magnitudes: th.Tensor,
    window_size: int = 0,
    padding: Text = "same",
) -> th.Tensor:
    """Filter audio with a finite impulse response filter.

    Args:
        audio: Input audio. Tensor of shape [batch_size, audio_timesteps].
            magnitudes: Frequency transfer curve. Float32 Tensor of shape
            [batch_size, n_frames, n_frequencies] or [batch_size, n_frequencies].
            The frequencies of the last dimension are ordered as
            [0, f_nyqist / (n_frames -1), ..., f_nyquist], where f_nyquist is
            (sample_rate / 2). Automatically splits the audio into equally
            sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If
            window_size is less than 1, it is set as the default
            (n_frequencies).
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the
            audio is extended to include the tail of the impulse response
            (audio_timesteps + window_size - 1).

    Returns:
        Filtered audio.
            Shape [batch_size, audio_timesteps + window_size - 1] ('valid'
            padding) or shape [batch_size, audio_timesteps] ('same' padding).
    """
    impulse_response = frequency_impulse_response(
        magnitudes, window_size=window_size
    )
    return fft_convolve(audio, impulse_response, padding=padding)


def sinc_filter(
    audio: th.Tensor,
    cutoff_frequency: th.Tensor,
    window_size: int = 512,
    sample_rate: int = None,
    padding: Text = "same",
) -> th.Tensor:
    """Filter audio with sinc low-pass filter.

    Args:
        audio: Input audio. Tensor of shape [batch_size, audio_timesteps].
        cutoff_frequency: Frequency cutoff for low-pass sinc filter. If the
            sample_rate is given, cutoff_frequency is in Hertz. If sample_rate
            is None, cutoff_frequency is normalized ratio (frequency/nyquist)
            in the range [0, 1.0]. Shape [batch_size, n_time, 1].
        window_size: Size of the Hamming window to apply to the impulse.
        sample_rate: Optionally provide the sample rate.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the
            audio is extended to include the tail of the impulse response
            (audio_timesteps + window_size - 1).

    Returns:
        Filtered audio. Tensor of shape
        [batch_size, audio_timesteps + window_size - 1] ('valid' padding) or shape
        [batch_size, audio_timesteps] ('same' padding).
    """
    impulse_response = sinc_impulse_response(
        cutoff_frequency, window_size=window_size, sample_rate=sample_rate
    )
    return fft_convolve(audio, impulse_response, padding=padding)
