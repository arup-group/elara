import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../elara")))
from elara.config import Config
from elara import benchmarking
from elara.config import PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.event_handlers import EventHandlerWorkStation
from elara.plan_handlers import PlanHandlerWorkStation

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
config = Config(config_path)


@pytest.mark.skip(reason=None)
def test_town_hourly_in_cordon_score_zero():
    benchmark = benchmarking.TestTownHourlyCordon
    test_bm = benchmark(
        config,
        'car',
    )
    score = test_bm.build({})
    assert score['in'] == 0


@pytest.mark.skip(reason=None)
def test_town_hourly_out_cordon_score_zero():
    benchmark = benchmarking.TestTownHourlyCordon
    test_bm = benchmark(
        config,
        'car',
    )
    score = test_bm.build({})
    assert score['out'] == 0


@pytest.mark.skip(reason=None)
def test_town_peak_in_cordon_score_zero():
    benchmark = benchmarking.TestTownPeakIn
    test_bm = benchmark(
        config,
        'car',
    )
    score = test_bm.build({})
    assert score['in'] == 0


@pytest.mark.skip(reason=None)
def test_town_mode_share_score_zero():
    benchmark = benchmarking.TestTownCommuterStats
    test_bm = benchmark(
        config,
        'all',
    )
    score = test_bm.build({})
    assert score['modeshare'] == 0


# Config
@pytest.fixture
def test_config():
    config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
    config = Config(config_path)
    assert config
    return config


# Paths
@pytest.fixture
def test_paths(test_config):
    paths = PathFinderWorkStation(test_config)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths


@pytest.mark.skip(reason=None)
def test_benchmark_workstation(test_config, test_paths):
    input_workstation = InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    event_workstation = EventHandlerWorkStation(test_config)
    event_workstation.connect(managers=None, suppliers=[input_workstation])
    event_workstation.load_all_tools(option='bus')
    event_workstation.build()

    plan_workstation = PlanHandlerWorkStation(test_config)
    plan_workstation.connect(managers=None, suppliers=[input_workstation])
    tool = plan_workstation.tools['mode_share']
    plan_workstation.resources['mode_share'] = tool(test_config, 'all')
    plan_workstation.build()

    pp_workstation = benchmarking.BenchmarkWorkStation(test_config)
    pp_workstation.connect(managers=None, suppliers=[event_workstation, plan_workstation])
    pp_workstation.resources['test_town_cordon:car'] = pp_workstation.tools['test_town_cordon'](
        test_config, 'car')
    pp_workstation.build()

