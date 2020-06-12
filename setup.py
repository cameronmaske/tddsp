#!/usr/bin/env python
import os
import sys
import setuptools

version_path = os.path.join(os.path.dirname(__file__), "tddsp")
sys.path.append(version_path)
from version import __version__


setuptools.setup(
    name="tddsp",
    version=__version__,
    description="Differentiable Digital Signal Processing for pytorch",
    author="Sven Rodriguez",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "torch"],
    keywords="audio dsp signalprocessing machinelearning music",
)
