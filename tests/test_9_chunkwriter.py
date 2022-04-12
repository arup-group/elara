import sys
import os
import pytest

sys.path.append(os.path.abspath('../elara'))
from elara.factory import CSVChunkWriter, ArrowChunkWriter

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
test_inputs = os.path.join(test_dir, "test_intermediate_data")
test_outputs = os.path.join(test_dir, "test_outputs")
if not os.path.exists(test_outputs):
    os.mkdir(test_outputs)
benchmarks_path = os.path.join(test_outputs, "benchmarks")
if not os.path.exists(benchmarks_path):
    os.mkdir(benchmarks_path)


@pytest.fixture
def csv_data_streamer():
    line = {'a': 1, 'b': 2}
    return [line for _ in range(10)]


def test_add(csv_data_streamer):
    writer = CSVChunkWriter(os.path.join(test_outputs, "test_chunks.csv"), chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(csv_data_streamer)
    assert len(writer.chunk) == 10


def test_write(csv_data_streamer):
    writer = CSVChunkWriter(os.path.join(test_outputs, "test_chunks.csv"), chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(csv_data_streamer)
    assert len(writer.chunk) == 10
    writer.write()
    assert len(writer.chunk) == 0


def test_auto_write(csv_data_streamer):
    writer = CSVChunkWriter(os.path.join(test_outputs, "test_chunks.csv"), chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(csv_data_streamer)
    assert len(writer.chunk) == 10
    writer.add(csv_data_streamer)
    assert len(writer.chunk) == 0


def test_auto_write_twice(csv_data_streamer):
    writer = CSVChunkWriter(os.path.join(test_outputs, "test_chunks.csv"), chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(csv_data_streamer)
    writer.add(csv_data_streamer)
    writer.add(csv_data_streamer)
    writer.add(csv_data_streamer)
    assert len(writer.chunk) == 0


@pytest.fixture
def arrow_data_A():
    return [{
        "trip": 0,
        "pid": "0_0",
        "mode": "car",
        "path": [[-1.960884, 52.667599], [-1.960884, 52.667599],[-1.960884, 52.667599], [-1.960884, 52.667599]],
        "timestamps": [1590083729, 1590083729, 1590083729, 1590083729],
        "color": [25, 66, 102]
    }]


@pytest.fixture
def arrow_data_B():
    return [{
        "trip": 1,
        "pid": "1_0",
        "mode": "bus",
        "path": [[-1.960884, 52.667599], [-1.960884, 52.667599]],
        "timestamps": [1590083729, 1590083729],
        "color": [25, 66, 10]
    }]


def test_add_arrow(arrow_data_A, arrow_data_B):
    writer = ArrowChunkWriter(os.path.join(test_outputs, "test_chunks.arrow"), chunksize=15)
    assert len(writer.chunk) == 0
    writer.add(arrow_data_A)
    assert len(writer.chunk) == 1
    writer.add(arrow_data_B)
    assert len(writer.chunk) == 2


def test_arrow_write(arrow_data_A, arrow_data_B):
    writer = ArrowChunkWriter(os.path.join(test_outputs, "test_chunks.arrow"), chunksize=15)
    writer.add(arrow_data_A)
    writer.add(arrow_data_B)
    writer.write()
    assert len(writer.chunk) == 0


def test_arrow_auto_write(arrow_data_A, arrow_data_B):
    writer = ArrowChunkWriter(os.path.join(test_outputs, "test_chunks.arrow"), chunksize=2)
    writer.add(arrow_data_A)
    assert len(writer.chunk) == 1
    writer.add(arrow_data_B)
    assert len(writer.chunk) == 0


def test_arrow_auto_write_twice(arrow_data_A, arrow_data_B):
    writer = ArrowChunkWriter(os.path.join(test_outputs, "test_chunks.arrow"), chunksize=2)
    writer.add(arrow_data_A)
    writer.add(arrow_data_B)
    writer.add(arrow_data_A)
    writer.add(arrow_data_B)
    assert len(writer.chunk) == 0