import sys
import os
import pytest

# paths in config files etc. assume we're in the repo's root, so make sure we always are
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root_dir)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../elara")))
from elara.config import Config
from elara import benchmarking
from elara.config import PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.event_handlers import EventHandlerWorkStation
from elara.plan_handlers import PlanHandlerWorkStation

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
test_inputs = os.path.join(test_dir, "test_intermediate_data")
test_outputs = os.path.join(test_dir, "test_outputs")
if not os.path.exists(test_outputs):
    os.mkdir(test_outputs)
benchmarks_path = os.path.join(test_outputs, "benchmarks")
if not os.path.exists(benchmarks_path):
    os.mkdir(benchmarks_path)

config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
config = Config(config_path)


def test_pt_volume_counter_bus():
    benchmark = benchmarking.TestPTVolume
    test_bm = benchmark(
        config,
        'bus',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0
    

def test_pt_interactions_counter_bus():
    benchmark = benchmarking.TestPTInteraction
    test_bm = benchmark(
        config,
        'bus',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0


def test_links_counter_car_init():
    benchmark = benchmarking.TestCordon
    test_bm = benchmark(
        config,
        'car',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0


def test_links_counter_bus_init():
    benchmark = benchmarking.TestCordon
    test_bm = benchmark(
        config,
        'bus',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0


def test_points_counter_init():
    benchmark = benchmarking.TestHighwayCounters
    test_bm = benchmark(
        config,
        'car',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0


def test_town_hourly_in_cordon_score_zero():
    benchmark = benchmarking.TestTownHourlyCordon
    test_bm = benchmark(
        config,
        'car',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['in'] == 0


def test_town_hourly_out_cordon_score_zero():
    benchmark = benchmarking.TestTownHourlyCordon
    test_bm = benchmark(
        config,
        'car',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['out'] == 0


def test_town_peak_in_cordon_score_zero():
    benchmark = benchmarking.TestTownPeakIn
    test_bm = benchmark(
        config,
        'car',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['in'] == 0


def test_town_mode_share_score_zero():
    benchmark = benchmarking.TestTownCommuterStats
    test_bm = benchmark(
        config,
        'all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0


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


def test_benchmark_workstation(test_config, test_paths):
    input_workstation = InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    event_workstation = EventHandlerWorkStation(test_config)
    event_workstation.connect(managers=None, suppliers=[input_workstation])
    event_workstation.load_all_tools(option='bus')
    event_workstation.build(write_path=test_outputs)

    plan_workstation = PlanHandlerWorkStation(test_config)
    plan_workstation.connect(managers=None, suppliers=[input_workstation])
    tool = plan_workstation.tools['mode_share']
    plan_workstation.resources['mode_share'] = tool(test_config, 'all')
    plan_workstation.build(write_path=test_outputs)

    pp_workstation = benchmarking.BenchmarkWorkStation(test_config)
    pp_workstation.connect(managers=None, suppliers=[event_workstation, plan_workstation])
    pp_workstation.resources['test_town_cordon:car'] = pp_workstation.tools['test_town_cordon'](
        test_config, 'car')
    pp_workstation.build(write_path=test_outputs)


def test_benchmark_workstation_with_link_bms(test_config, test_paths):
    input_workstation = InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    event_workstation = EventHandlerWorkStation(test_config)
    event_workstation.connect(managers=None, suppliers=[input_workstation])
    tool = event_workstation.tools['volume_counts']
    event_workstation.resources['volume_counts'] = tool(test_config, 'all')
    event_workstation.build(write_path=test_outputs)

    plan_workstation = PlanHandlerWorkStation(test_config)
    plan_workstation.connect(managers=None, suppliers=[input_workstation])
    tool = plan_workstation.tools['mode_share']
    plan_workstation.resources['mode_share'] = tool(test_config, 'all')
    plan_workstation.build(write_path=test_outputs)

    pp_workstation = benchmarking.BenchmarkWorkStation(test_config)
    pp_workstation.connect(managers=None, suppliers=[event_workstation, plan_workstation])

    pp_workstation.resources['test_link_cordon:car'] = pp_workstation.tools['test_link_cordon'](
        test_config, 'car')

    pp_workstation.resources['test_town_cordon:car'] = pp_workstation.tools['test_town_cordon'](
        test_config, 'car')
        
    pp_workstation.build(write_path=test_outputs)


def test_all_paths_exist(test_config):
    benchmark_workstation = benchmarking.BenchmarkWorkStation(test_config)
    for name, tool in benchmark_workstation.tools.items():
        try:
            assert os.path.exists(tool.benchmark_data_path)
        except AttributeError:
            continue
