import sys
import os
import pytest

sys.path.append(os.path.abspath('../elara'))
from elara.factory import ChunkWriter

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
test_inputs = os.path.join(test_dir, "test_intermediate_data")
test_outputs = os.path.join(test_dir, "test_outputs")
if not os.path.exists(test_outputs):
    os.mkdir(test_outputs)
benchmarks_path = os.path.join(test_outputs, "benchmarks")
if not os.path.exists(benchmarks_path):
    os.mkdir(benchmarks_path)


@pytest.fixture
def test_data_streamer():
    line = {'a': 1, 'b': 2}
    return [line for _ in range(10)]


def test_add(test_data_streamer):
    writer = ChunkWriter(os.path.join(test_outputs, "test_chunks.csv"), chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(test_data_streamer)
    assert len(writer.chunk) == 10


def test_write(test_data_streamer):
    writer = ChunkWriter(os.path.join(test_outputs, "test_chunks.csv"), chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(test_data_streamer)
    assert len(writer.chunk) == 10
    writer.write()
    assert len(writer.chunk) == 0


def test_auto_write(test_data_streamer):
    writer = ChunkWriter(os.path.join(test_outputs, "test_chunks.csv"), chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(test_data_streamer)
    assert len(writer.chunk) == 10
    writer.add(test_data_streamer)
    assert len(writer.chunk) == 0

