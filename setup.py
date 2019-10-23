"""Packaging settings."""

from setuptools import find_packages, setup

from elara import __version__


setup(
    name="elara",
    version=__version__,
    description="A command line tool for processing MATSim events.",
    packages=find_packages(),  # exclude="tests*"
    install_requires=[
        "geopandas",
        "halo",
        "lxml",
        "pandas",
        "toml",
    ],
    entry_points={"console_scripts": ["elara = elara.main:cli"]},
)
