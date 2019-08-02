import sys
import os
import pytest

sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import benchmarking
sys.path.append(os.path.abspath('../tests'))

config_path = os.path.join('tests/test_xml_scenario.toml')
config = Config(config_path)


def test_town_hourly_in_cordon_score_zero():
    benchmark = benchmarking.TestTownHourlyCordon
    test_bm = benchmark(
        'test_cordon', config
    )
    score = test_bm.output_and_score()
    assert score['in'] == 0


def test_town_hourly_out_cordon_score_zero():
    benchmark = benchmarking.TestTownHourlyCordon
    test_bm = benchmark(
        'test_cordon', config
    )
    score = test_bm.output_and_score()
    assert score['out'] == 0


def test_town_peak_in_cordon_score_zero():
    benchmark = benchmarking.TestTownPeakIn
    test_bm = benchmark(
        'test_cordon', config
    )
    score = test_bm.output_and_score()
    assert score['in'] == 0


def test_town_mode_share_score_zero():
    benchmark = benchmarking.TestTownCommuterStats
    test_bm = benchmark(
        'test_cordon', config
    )
    score = test_bm.output_and_score()
    assert score['modeshare'] == 0

