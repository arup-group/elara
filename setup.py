"""Packaging settings."""
import os

from setuptools import find_packages, setup

from elara import __version__

requirementPath="requirements.txt"
install_requires = [] # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name="elara",
    version=__version__,
    description="A command line tool for processing MATSim events.",
    packages=find_packages(),  # exclude="tests*"
    install_requires=install_requires,
    entry_points={"console_scripts": ["elara = elara.main:cli"]},
)
