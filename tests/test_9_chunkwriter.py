import sys
import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree


sys.path.append(os.path.abspath('../elara'))
from elara.factory import ChunkWriter
sys.path.append(os.path.abspath('../tests'))


@pytest.fixture
def test_data_streamer():
    line = {'a': 1, 'b':2}
    return [line for _ in range(10)]


def test_add(test_data_streamer):
    writer = ChunkWriter("./test_chunks.csv", chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(test_data_streamer)
    assert len(writer.chunk) == 10


def test_write(test_data_streamer):
    writer = ChunkWriter("./test_chunks.csv", chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(test_data_streamer)
    assert len(writer.chunk) == 10
    writer.write()
    assert len(writer.chunk) == 0


def test_auto_write(test_data_streamer):
    writer = ChunkWriter("./test_chunks.csv", chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(test_data_streamer)
    assert len(writer.chunk) == 10
    writer.add(test_data_streamer)
    assert len(writer.chunk) == 0

