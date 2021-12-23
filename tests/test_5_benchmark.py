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

def test_trip_duration_comparison_cars():
    benchmark = benchmarking.TestTripDurationComparisonCars(
        config=config,
        mode="all"
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_trip_duration_comparison_all():
    benchmark = benchmarking.TestTripDurationComparisonAll(
        config=config,
        mode="all"
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_trip_duration_comparison_mode_consistency():
    benchmark = benchmarking.TestTripDurationComparisonWithModeConsistency(
        config=config,
        mode="all",
        mode_consistent=True
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_link_vehicle_speed_comparison():
    benchmark = benchmarking.TestLinkVehicleSpeedsComparison(
        config=config,
        mode="car",
        time_slice=8
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_link_vehicle_speed_comparison_with_groupby_subpopulation():
    benchmark = benchmarking.TestLinkVehicleSpeedsComparisonByAttributeComparison(
        config=config,
        mode="car",
        time_slice=8,
        groupby_person_attribute="subpopulation"
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


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


def test_mode_share_score_zero():
    benchmark = benchmarking.TestTripModeSharesComparison
    test_bm = benchmark(
        config,
        'all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_mode_share_by_attribute_score_zero():
    benchmark = benchmarking.TestTripModeSharesByAttributeComparison
    test_bm = benchmark(
        config,
        mode='all',
        groupby_person_attribute="subpopulation",
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_mode_counts_score_zero():
    benchmark = benchmarking.TestTripModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_mode_counts_by_attribute_score_zero():
    benchmark = benchmarking.TestTripModeCountsByAttributeComparison
    test_bm = benchmark(
        config,
        mode='all',
        groupby_person_attribute="subpopulation"
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_activity_mode_share_score_zero():
    benchmark = benchmarking.TestTripActivityModeSharesComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_activity_subpopulaion_mode_share_score_zero():
    benchmark = benchmarking.TestTripActivityModeSharesByAttributeComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        groupby_person_attribute="subpopulation",
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_activity_mode_share_count_zero():
    benchmark = benchmarking.TestTripActivityModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_activity_subpopulaion_mode_count_score_zero():
    benchmark = benchmarking.TestTripActivityModeCountsByAttributeComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        groupby_person_attribute="subpopulation",
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_mode_share_score_zero():
    benchmark = benchmarking.TestPlanModeSharesComparison
    test_bm = benchmark(
        config,
        'all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_mode_share_by_attribute_score_zero():
    benchmark = benchmarking.TestPlanModeSharesByAttributeComparison
    test_bm = benchmark(
        config,
        mode='all',
        groupby_person_attribute="subpopulation",
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_mode_counts_score_zero():
    benchmark = benchmarking.TestPlanModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_mode_counts_by_attribute_score_zero():
    benchmark = benchmarking.TestPlanModeCountsByAttributeComparison
    test_bm = benchmark(
        config,
        mode='all',
        groupby_person_attribute="subpopulation"
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_activity_mode_share_score_zero():
    benchmark = benchmarking.TestPlanActivityModeSharesComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_plan_activity_subpopulaion_mode_share_score_zero():
    benchmark = benchmarking.TestPlanActivityModeSharesByAttributeComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        groupby_person_attribute="subpopulation",
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_activity_mode_share_count_zero():
    benchmark = benchmarking.TestPlanActivityModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_activity_subpopulaion_mode_count_score_zero():
    benchmark = benchmarking.TestPlanActivityModeCountsByAttributeComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        groupby_person_attribute="subpopulation",
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_duration_comparison_score_zero():
    benchmark = benchmarking.TestDurationBreakdownComparison
    test_bm = benchmark(
        config,
        'all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_duration_comparison_score_zero():
    benchmark = benchmarking.TestDurationBreakdownComparison
    test_bm = benchmark(
        config,
        'all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_duration_comparison_score_zero_filepath():
    benchmark = benchmarking.DurationBreakdownComparison
    benchmark_data_path = os.path.join('tests','test_outputs','trip_duration_breakdown_all.csv')
    test_bm = benchmark(
        config,
        'all',
        benchmark_data_path = benchmark_data_path
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


# Config
@pytest.fixture
def test_config():
    config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
    config = Config(config_path)
    assert config
    return config

@pytest.fixture
def test_config_dictionary():
    config_dictionary_path = os.path.join(test_dir, 'test_xml_scenario_dictionary.toml')
    config_dictionary = Config(config_dictionary_path)
    assert config_dictionary
    return config_dictionary


# Paths
@pytest.fixture
def test_paths(test_config):
    paths = PathFinderWorkStation(test_config)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths


def test_benchmark_workstation_with_link_bms(test_config, test_paths):
    input_workstation = InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    event_workstation = EventHandlerWorkStation(test_config)
    event_workstation.connect(managers=None, suppliers=[input_workstation])
    tool = event_workstation.tools['link_vehicle_counts']
    event_workstation.resources['link_vehicle_counts'] = tool(test_config, 'all')
    event_workstation.build(write_path=test_outputs)

    plan_workstation = PlanHandlerWorkStation(test_config)
    plan_workstation.connect(managers=None, suppliers=[input_workstation])
    tool = plan_workstation.tools['trip_modes']
    plan_workstation.resources['trip_modes'] = tool(test_config, 'all')
    plan_workstation.build(write_path=test_outputs)

    pp_workstation = benchmarking.BenchmarkWorkStation(test_config)
    pp_workstation.connect(managers=None, suppliers=[event_workstation, plan_workstation])

    pp_workstation.resources['test_link_cordon:car'] = pp_workstation.tools['test_link_cordon'](
        test_config, 'car')

    pp_workstation.build(write_path=test_outputs)


def test_benchmark_duration_workstation_dictionary(test_config_dictionary, test_paths):
    # call duration comparison handler with dictionary arguments
    input_workstation = InputsWorkStation(test_config_dictionary)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    event_workstation = EventHandlerWorkStation(test_config_dictionary)
    event_workstation.connect(managers=None, suppliers=[input_workstation])

    plan_workstation = PlanHandlerWorkStation(test_config_dictionary)
    plan_workstation.connect(managers=None, suppliers=[input_workstation])

    pp_workstation = benchmarking.BenchmarkWorkStation(test_config_dictionary)
    pp_workstation.connect(managers=None, suppliers=[event_workstation, plan_workstation])

    benchmarking_workstation = benchmarking.BenchmarkWorkStation(test_config_dictionary)
    benchmarking_workstation.connect(managers=None, suppliers=[event_workstation, plan_workstation, pp_workstation])
    tool = benchmarking_workstation.tools['duration_breakdown_comparison']
    benchmarking_workstation.resources['duration_breakown_comparison'] = tool(test_config_dictionary, 'all', benchmark_data_path='./tests/test_outputs/trip_duration_breakdown_all.csv')

    benchmarking_workstation.build(write_path=test_outputs)