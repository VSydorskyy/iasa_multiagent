import os
from configparser import ConfigParser

from setuptools import find_packages, setup

config = ConfigParser()
rc = os.path.join(os.path.expanduser("~"), ".pypirc")
config.read(rc)

setup(
    name="matk",
    version="0.0.1",
    description=("Different Multiagnet System models"),
    author="Vladimir Sydorskyi",
    python_requires=">=3.7",
    install_requires=[
        "tqdm==4.48.0",
        "matplotlib==3.3.0",
        "numpy==1.17.2",
        "scikit-learn>=0.22.2",
    ],
    packages=find_packages(),
)
