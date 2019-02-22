import sys
import os

sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import benchmarking
sys.path.append(os.path.abspath('../tests'))


# Benchmark
def test_inner_cordon_init():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    sys.path.append(os.path.abspath('../elara'))
    inner_cordon_benchmark = benchmarking.InnerCordon(
        'inner_cordon', config
    )
    assert inner_cordon_benchmark
    sys.path.append(os.path.abspath('../tests'))


def test_all_benchmark_paths_exist():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    sys.path.append(os.path.abspath('../elara'))
    for benchmark_name, benchmark in benchmarking.BENCHMARK_MAP.items():
        benchmarker = benchmark(
            benchmark_name, config
        )
        assert os.path.exists(benchmarker.benchmark_path)
        assert os.path.exists(benchmarker.map_path)

    sys.path.append(os.path.abspath('../tests'))


def test_inner_cordon_scoring():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    sys.path.append(os.path.abspath('../elara'))
    inner_cordon_benchmark = benchmarking.InnerCordon(
        'inner_cordon', config
    )
    score = inner_cordon_benchmark.output_and_score()
    assert score

    sys.path.append(os.path.abspath('../tests'))