# TDDSP: (py)Torch Differentiable Digital Signal Processing

**WARNING: This Repository is NOT the official implementation.**

Re-implementation of [_Magenta's DDSP_](https://github.com/magenta/ddsp) library for pytorch. Code is adapted directly from the tensorflow version.

>Original Authors : Jesse Engel, Lamtharn (Hanoi) Hantrakul, Chenjie Gu, Adam Roberts (Google), https://openreview.net/pdf?id=B1x1ma4tDr


## Getting Started

The lib works the same way as the official repo, but with torch Tensors, and torchaudio's [dimensions conventions](https://github.com/pytorch/audio#conventions).

```
import tddsp

# Get synthesizer parameters from a neural network.
outputs = network(inputs)

# Initialize signal processors.
additive = tddsp.synths.Additive()

# Generates audio from additive synthesizer.
audio = additive(outputs['amplitudes'],
                 outputs['harmonic_distribution'],
                 outputs['f0_hz'])
```

## Installation

Requires torch >= 1.5.0 and torchaudio >= 0.5.0. It might work with lower versions, I haven't tested it.

```
pip install git+https://github.com/svenrdz/tddsp
```

You can run all unit tests using `pytest`.
