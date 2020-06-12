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
"""Library of FFT operations for loss functions and conditioning."""

# import crepe
from tddsp.core import f32, complex_abs
import librosa
import numpy as np
import torch as th
import torchaudio as tha

_CREPE_SAMPLE_RATE = 16000
_CREPE_FRAME_SIZE = 1024

F0_RANGE = 127.0  # MIDI
LD_RANGE = 120.0  # dB

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def safe_log(x, eps=1e-5):
    return (x + eps).log()


def mel_to_hertz(mel_values):
    """Converts frequencies in `mel_values` from the mel scale to linear scale."""
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        (f32(mel_values) / _MEL_HIGH_FREQUENCY_Q).exp() - 1.0
    )


def hertz_to_mel(frequencies_hertz):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
    return (
        _MEL_HIGH_FREQUENCY_Q
        * (1.0 + (f32(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ)).log()
    )


def linear_to_mel_weight_matrix(
    num_mel_bins=20,
    num_spectrogram_bins=129,
    sample_rate=16000,
    lower_edge_hertz=125.0,
    upper_edge_hertz=3800.0,
):
    """Returns a matrix to warp linear scale spectrograms to the mel scale.

    Adapted from tf.contrib.signal.linear_to_mel_weight_matrix with a minimum
    band width (in Hz scale) of 1.5 * freq_bin. To preserve accuracy,
    we compute the matrix at float64 precision and then cast to `dtype`
    at the end. This function can be constant folded by graph optimization
    since there are no Tensor inputs.

    Args:
        num_mel_bins: Int, number of output frequency dimensions.
        num_spectrogram_bins: Int, number of input frequency dimensions.
        sample_rate: Int, sample rate of the audio.
        lower_edge_hertz: Float, lowest frequency to consider.
        upper_edge_hertz: Float, highest frequency to consider.

    Returns:
        Numpy float32 matrix of shape [num_spectrogram_bins, num_mel_bins].

    Raises:
        ValueError: Input argument in the wrong range.
    """
    # Validate input arguments
    if num_mel_bins <= 0:
        raise ValueError(
            "num_mel_bins must be positive. Got: %s" % num_mel_bins
        )
    if num_spectrogram_bins <= 0:
        raise ValueError(
            "num_spectrogram_bins must be positive. Got: %s"
            % num_spectrogram_bins
        )
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive. Got: %s" % sample_rate)
    if lower_edge_hertz < 0.0:
        raise ValueError(
            "lower_edge_hertz must be non-negative. Got: %s" % lower_edge_hertz
        )
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(
            "lower_edge_hertz %.1f >= upper_edge_hertz %.1f"
            % (lower_edge_hertz, upper_edge_hertz)
        )
    if upper_edge_hertz > sample_rate / 2:
        raise ValueError(
            "upper_edge_hertz must not be larger than the Nyquist "
            "frequency (sample_rate / 2). Got: %s for sample_rate: %s"
            % (upper_edge_hertz, sample_rate)
        )

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = th.linspace(
        0.0, nyquist_hertz, num_spectrogram_bins, dtype=th.float64
    )[None, bands_to_zero:]
    # spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = th.linspace(
        hertz_to_mel(lower_edge_hertz),
        hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2,
        dtype=th.float64,
    )

    lower_edge_mel = band_edges_mel[0:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]

    freq_res = nyquist_hertz / float(num_spectrogram_bins)
    freq_th = 1.5 * freq_res
    for i in range(0, num_mel_bins):
        center_hz = mel_to_hertz(center_mel[i])
        lower_hz = mel_to_hertz(lower_edge_mel[i])
        upper_hz = mel_to_hertz(upper_edge_mel[i])
        if upper_hz - lower_hz < freq_th:
            rhs = 0.5 * freq_th / (center_hz + _MEL_BREAK_FREQUENCY_HERTZ)
            dm = _MEL_HIGH_FREQUENCY_Q * (rhs + (1.0 + rhs ** 2).sqrt()).log()
            lower_edge_mel[i] = center_mel[i] - dm
            upper_edge_mel[i] = center_mel[i] + dm

    lower_edge_hz = mel_to_hertz(lower_edge_mel)[:, None]
    center_hz = mel_to_hertz(center_mel)[:, None]
    upper_edge_hz = mel_to_hertz(upper_edge_mel)[:, None]

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (linear_frequencies - lower_edge_hz) / (
        center_hz - lower_edge_hz
    )
    upper_slopes = (upper_edge_hz - linear_frequencies) / (
        upper_edge_hz - center_hz
    )

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = th.max(
        th.zeros_like(lower_slopes), th.min(lower_slopes, upper_slopes)
    )

    # Re-add the zeroed lower bins we sliced out above.
    # [freq, mel]
    mel_weights_matrix = th.nn.functional.pad(
        mel_weights_matrix, (bands_to_zero, 0), "constant"
    )
    return mel_weights_matrix.type(th.float32)


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Differentiable stft in torch, computed in batch."""
    audio = f32(audio)
    assert frame_size * overlap % 2.0 == 0.0
    hop_size = int(frame_size * (1.0 - overlap))
    is_2d = len(audio.shape) == 2

    if pad_end:
        n_samples_initial = audio.shape[-1]
        n_frames = int(np.ceil(n_samples_initial / hop_size))
        n_samples_final = (n_frames - 1) * hop_size + frame_size
        pad = n_samples_final - n_samples_initial
        padding = (0, pad)
        audio = audio[None, ...] if not is_2d else audio
        audio = th.nn.functional.pad(audio, padding, "constant")
        audio = audio[0] if not is_2d else audio

    s = th.stft(
        audio,
        window=th.hann_window(int(frame_size)),
        hop_length=hop_size,
        n_fft=int(frame_size),
        center=False,
    )
    return s


def stft_np(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Non-differentiable stft using librosa, one example at a time."""
    assert frame_size * overlap % 2.0 == 0.0
    hop_size = int(frame_size * (1.0 - overlap))
    is_2d = len(audio.shape) == 2

    if pad_end:
        n_samples_initial = int(audio.shape[-1])
        n_frames = int(np.ceil(n_samples_initial / hop_size))
        n_samples_final = (n_frames - 1) * hop_size + frame_size
        pad = n_samples_final - n_samples_initial
        padding = ((0, 0), (0, pad))
        audio = audio[None, ...] if not is_2d else audio
        audio = np.pad(audio, padding, "constant")
        audio = audio[0] if not is_2d else audio

    def stft_fn(y):
        return librosa.stft(
            y=y, n_fft=int(frame_size), hop_length=hop_size, center=False
        )

    s = np.stack([stft_fn(a) for a in audio]) if is_2d else stft_fn(audio)
    return s


def compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
    mag = stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end)
    return complex_abs(mag)


def compute_mel(
    audio,
    lo_hz=0.0,
    hi_hz=8000.0,
    bins=64,
    fft_size=2048,
    sample_rate=16000,
    overlap=0.75,
    pad_end=True,
    use_th=True,
):
    """Calculate Mel Spectrogram."""
    mag = compute_mag(audio, fft_size, overlap, pad_end)
    num_spectrogram_bins = mag.shape[0]
    linear_to_mel_matrix = linear_to_mel_weight_matrix(
        bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz
    )
    return th.tensordot(linear_to_mel_matrix, mag, 1)


def compute_logmag(audio, size=2048, overlap=0.75, pad_end=True):
    return safe_log(compute_mag(audio, size, overlap, pad_end))


def compute_logmel(
    audio,
    lo_hz=80.0,
    hi_hz=7600.0,
    bins=64,
    fft_size=2048,
    sample_rate=16000,
    overlap=0.75,
    pad_end=True,
):
    mel = compute_mel(
        audio, lo_hz, hi_hz, bins, fft_size, sample_rate, overlap, pad_end
    )
    return safe_log(mel)


def compute_mfcc(
    audio,
    lo_hz=20.0,
    hi_hz=8000.0,
    fft_size=1024,
    mel_bins=128,
    mfcc_bins=13,
    sample_rate=16000,
    overlap=0.75,
    pad_end=True,
):
    """Calculate Mel-frequency Cepstral Coefficients."""
    logmel = compute_logmel(
        audio,
        lo_hz=lo_hz,
        hi_hz=hi_hz,
        bins=mel_bins,
        fft_size=fft_size,
        sample_rate=sample_rate,
        overlap=overlap,
        pad_end=pad_end,
    )
    dct = tha.functional.create_dct(
        n_mfcc=mfcc_bins, n_mels=mel_bins, norm=None
    )
    mfccs = dct.T @ logmel
    return mfccs[:mfcc_bins]


def diff(x, axis=-1):
    """Take the finite difference of a tensor along an axis.

    Args:
        x: Input tensor of any dimension.
        axis: Axis on which to take the finite difference.

    Returns:
        d: Tensor with size less than x by 1 along the difference dimension.

    Raises:
        ValueError: Axis out of range for tensor.
    """
    if axis >= len(x.shape):
        raise ValueError(
            "Invalid axis index: %d for tensor with only %d axes."
            % (axis, len(x.shape))
        )

    slice_front = [
        slice(0 + int(i == axis), size) for i, size in enumerate(x.shape)
    ]
    slice_back = [
        slice(0, size - int(i == axis)) for i, size in enumerate(x.shape)
    ]
    return x[slice_front] - x[slice_back]


def compute_loudness(
    audio,
    sample_rate=16000,
    frame_rate=250,
    n_fft=2048,
    range_db=LD_RANGE,
    ref_db=20.7,
    use_th=False,
):
    """Perceptual loudness in dB, relative to white noise, amplitude=1.

    Function is differentiable if use_th=True.

    Args:
        audio: Numpy ndarray or tensor. Shape [batch_size, audio_length] or
            [batch_size,].
        sample_rate: Audio sample rate in Hz.
        frame_rate: Rate of loudness frames in Hz.
        n_fft: Fft window size.
        range_db: Sets the dynamic range of loudness in decibles. The minimum
        loudness (per a frequency bin) corresponds to -range_db.
        ref_db: Sets the reference maximum perceptual loudness as given by
            (A_weighting + 10 * log10(abs(stft(audio))**2.0). The default value
            corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a
            slight dependence on fft_size due to different granularity of perceptual
            weighting.
        use_th: Make function differentiable by using tensorflow.

    Returns:
        Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
    """
    if sample_rate % frame_rate != 0:
        raise ValueError(
            "frame_rate: {} must evenly divide sample_rate: {}."
            "For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz".format(
                frame_rate, sample_rate
            )
        )

    # Avoid log(0) instabilities.
    amin = 1e-20

    # Pick tensorflow or numpy.
    lib = th if use_th else np

    # Make inputs tensors for tensorflow.
    if use_th:
        audio, range_db, amin = f32(audio, range_db, amin)

    # Temporarily a batch dimension for single examples.
    is_1d = len(audio.shape) == 1
    audio = audio[None, :] if is_1d else audio

    # Take STFT.
    hop_size = sample_rate // frame_rate
    overlap = 1 - hop_size / n_fft
    stft_fn = stft if use_th else stft_np
    s = stft_fn(audio, frame_size=n_fft, overlap=overlap, pad_end=True)

    # Compute power
    amplitude = complex_abs(s) if use_th else np.abs(s)
    maximum = th.max if use_th else np.maximum
    power_db = lib.log10(maximum(amin, amplitude))
    power_db *= 20.0

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[None, :, None]
    a_weighting = f32(a_weighting) if use_th else a_weighting
    loudness = power_db + a_weighting

    # Set dynamic range.
    loudness -= ref_db
    loudness = maximum(-range_db, loudness)

    # Average over frequency bins.
    loudness = lib.mean(loudness, -2)

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    # Compute expected length of loudness vector
    n_secs = audio.shape[-1] / sample_rate  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)

    # Pad with `-range_db` noise floor or trim vector
    loudness = pad_or_trim_to_expected_length(
        loudness, expected_len, -range_db, use_th=use_th
    )
    return loudness


# def compute_f0(audio, sample_rate, frame_rate, viterbi=True):
#     """Fundamental frequency (f0) estimate using CREPE.

#     This function is non-differentiable and takes input as a numpy array.

#     Args:
#         audio: Numpy ndarray of single audio example. Shape [audio_length,].
#         sample_rate: Sample rate in Hz.
#         frame_rate: Rate of f0 frames in Hz.
#         viterbi: Use Viterbi decoding to estimate f0.

#     Returns:
#         f0_hz: Fundamental frequency in Hz. Shape [n_frames,].
#         f0_confidence: Confidence in Hz estimate (scaled [0, 1]). Shape [n_frames,].
#     """

#     n_secs = len(audio) / float(sample_rate)  # `n_secs` can have milliseconds
#     crepe_step_size = 1000 / frame_rate  # milliseconds
#     expected_len = int(n_secs * frame_rate)
#     audio = np.asarray(audio)

#     # Compute f0 with crepe.
#     _, f0_hz, f0_confidence, _ = crepe.predict(
#         audio,
#         sr=sample_rate,
#         viterbi=viterbi,
#         step_size=crepe_step_size,
#         center=False,
#         verbose=0,
#     )

#     # Postprocessing on f0_hz
#     f0_hz = pad_or_trim_to_expected_length(f0_hz, expected_len, 0)  # pad with 0
#     f0_hz = f0_hz.astype(np.float32)

#     # Postprocessing on f0_confidence
#     f0_confidence = pad_or_trim_to_expected_length(
#         f0_confidence, expected_len, 1
#     )
#     f0_confidence = np.nan_to_num(f0_confidence)  # Set nans to 0 in confidence
#     f0_confidence = f0_confidence.astype(np.float32)
#     return f0_hz, f0_confidence


def pad_or_trim_to_expected_length(
    vector, expected_len, pad_value=0, len_tolerance=20, use_th=False
):
    """Make vector equal to the expected length.

    Feature extraction functions like `compute_loudness()` or `compute_f0` produce
    feature vectors that vary in length depending on factors such as `sample_rate`
    or `hop_size`. This function corrects vectors to the expected length, warning
    the user if the difference between the vector and expected length was
    unusually high to begin with.

    Args:
        vector: Numpy 1D ndarray. Shape [vector_length,]
        expected_len: Expected length of vector.
        pad_value: Value to pad at end of vector.
        len_tolerance: Tolerance of difference between original and desired vector
            length.
        use_th: Make function differentiable by using tensorflow.

    Returns:
        vector: Vector with corrected length.

    Raises:
        ValueError: if `len(vector)` is different from `expected_len` beyond
            `len_tolerance` to begin with.
    """
    expected_len = int(expected_len)
    vector_len = int(vector.shape[-1])

    if abs(vector_len - expected_len) > len_tolerance:
        # Ensure vector was close to expected length to begin with
        raise ValueError(
            "Vector length: {} differs from expected length: {} "
            "beyond tolerance of : {}".format(
                vector_len, expected_len, len_tolerance
            )
        )

    is_1d = len(vector.shape) == 1
    vector = vector[None, :] if is_1d else vector

    # Pad missing samples
    if vector_len < expected_len:
        n_padding = expected_len - vector_len
        if use_th:
            vector = th.nn.functional.pad(
                f32(vector), (0, n_padding), mode="constant", value=pad_value
            )
        else:
            vector = np.pad(
                vector,
                ((0, 0), (0, n_padding)),
                mode="constant",
                constant_values=pad_value,
            )
    # Trim samples
    elif vector_len > expected_len:
        vector = vector[..., :expected_len]

    # Remove temporary batch dimension.
    vector = vector[0] if is_1d else vector
    return vector
