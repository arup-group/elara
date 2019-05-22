import sys
import os
import pytest

sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import benchmarking
sys.path.append(os.path.abspath('../tests'))


@pytest.mark.parametrize('benchmark_name',list(benchmarking.BENCHMARK_MAP.keys()))
def test_inner_cordon_scoring(benchmark_name):
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    benchmark = benchmarking.BENCHMARK_MAP.get(benchmark_name)
    test_bm = benchmark(
        benchmark_name, config
    )
    score = test_bm.output_and_score()
    assert score
