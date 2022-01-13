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
from elara import get_benchmark_data

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
    benchmark = benchmarking.TripDurationsComparison(
        config=config,
        mode="all",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'trip_durations_car.csv')
        )
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_trip_duration_comparison_all():
    benchmark = benchmarking.TripDurationsComparison(
        config=config,
        mode="all",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'trip_durations_multi_modal.csv')
        )
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_trip_duration_comparison_mode_consistency():
    benchmark = benchmarking.TripDurationsComparison(
        config=config,
        mode="all",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'trip_durations_mode_consistency.csv')
        ),
        mode_consistent=True
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_link_vehicle_speed_comparison():
    benchmark = benchmarking.LinkVehicleSpeedsComparison(
        config=config,
        mode="car",
        time_slice=8,
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'link_vehicle_speeds_car_average.csv')
        )
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_link_vehicle_speed_comparison_with_groupby_subpopulation():
    benchmark = benchmarking.LinkVehicleSpeedsComparison(
        config=config,
        mode="car",
        time_slice=8,
        groupby_person_attribute="subpopulation",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'link_vehicle_speeds_car_average_subpopulation.csv')
        )
    )
    score = benchmark.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_pt_volume_counter_bus():
    benchmark = benchmarking.PassengerStopToStop
    test_bm = benchmark(
        config,
        'bus',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_town', 'pt_stop_to_stop_volumes', 'test_pt_volumes_bus.json')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0
    

def test_pt_interactions_counter_bus():
    benchmark = benchmarking.TransitInteractionComparison
    test_bm = benchmark(
        config,
        'bus',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_town', 'pt_interactions', 'test_interaction_counter.json')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0


def test_links_counter_car_init():
    benchmark = benchmarking.LinkCounterComparison
    test_bm = benchmark(
        config,
        'car',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_town', 'test_town_cordon', 'test_link_counter.json')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0


def test_links_counter_bus_init():
    benchmark = benchmarking.LinkCounterComparison
    test_bm = benchmark(
        config,
        'bus',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_town', 'test_town_cordon', 'test_link_counter.json')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0


def test_points_counter_init():
    benchmark = benchmarking.PointsCounter
    test_bm = benchmark(
        config,
        'car',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_town', 'highways', 'test_hw_bm.json')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['counters'] == 0


def test_mode_share_score_zero():
    benchmark = benchmarking.TripModeSharesComparison
    test_bm = benchmark(
        config,
        'all',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'mode_shares.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_mode_share_by_attribute_score_zero():
    benchmark = benchmarking.TripModeSharesComparison
    test_bm = benchmark(
        config,
        mode='all',
        groupby_person_attribute="subpopulation",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'subpop_mode_shares.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_mode_counts_score_zero():
    benchmark = benchmarking.TripModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'mode_counts.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_mode_counts_by_attribute_score_zero():
    benchmark = benchmarking.TripModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        groupby_person_attribute="subpopulation",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'subpop_mode_counts.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_activity_mode_share_score_zero():
    benchmark = benchmarking.TripActivityModeSharesComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'mode_shares.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_activity_subpopulaion_mode_share_score_zero():
    benchmark = benchmarking.TripActivityModeSharesComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        groupby_person_attribute="subpopulation",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'subpop_mode_shares.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_activity_mode_share_count_zero():
    benchmark = benchmarking.TripActivityModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        benchmark_data_path=get_benchmark_data(
            os.path.join('test_fixtures', 'commuter_mode_counts.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_activity_subpopulaion_mode_count_score_zero():
    benchmark = benchmarking.TripActivityModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        groupby_person_attribute="subpopulation",
        benchmark_data_path=get_benchmark_data(
            os.path.join('test_fixtures', 'subpop_commuter_mode_counts.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_mode_share_score_zero():
    benchmark = benchmarking.PlanModeSharesComparison
    test_bm = benchmark(
        config,
        'all',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'mode_shares.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_mode_share_by_attribute_score_zero():
    benchmark = benchmarking.PlanModeSharesComparison
    test_bm = benchmark(
        config,
        mode='all',
        groupby_person_attribute="subpopulation",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'subpop_mode_shares.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_mode_counts_score_zero():
    benchmark = benchmarking.PlanModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'mode_plan_counts.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_mode_counts_by_attribute_score_zero():
    benchmark = benchmarking.PlanModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        groupby_person_attribute="subpopulation",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'subpop_mode_plan_counts.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_activity_mode_share_score_zero():
    benchmark = benchmarking.PlanActivityModeSharesComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'mode_shares.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_plan_activity_subpopulaion_mode_share_score_zero():
    benchmark = benchmarking.PlanActivityModeSharesComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        groupby_person_attribute="subpopulation",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'subpop_mode_shares.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_activity_mode_share_count_zero():
    benchmark = benchmarking.PlanActivityModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'commuter_plan_mode_counts.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_plan_activity_subpopulaion_mode_count_score_zero():
    benchmark = benchmarking.PlanActivityModeCountsComparison
    test_bm = benchmark(
        config,
        mode='all',
        destination_activity_filters=["work"],
        groupby_person_attribute="subpopulation",
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'subpop_commuter_plan_mode_counts.csv')
        )
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_input_plan_comparison_trip_start_zero():
    # input plan comparisons self-build benchmark_data_path, do not require separate test classes
    benchmark = benchmarking.InputPlanComparisonTripStart
    test_bm = benchmark(
        config,
        mode='all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_input_plan_comparison_trip_duration_zero():
    # input plan comparisons self-build benchmark_data_path, do not require separate test classes
    benchmark = benchmarking.InputPlanComparisonTripDuration
    test_bm = benchmark(
        config,
        mode='all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_input_plan_comparison_activity_start_zero():
    # input plan comparisons self-build benchmark_data_path, do not require separate test classes
    benchmark = benchmarking.InputPlanComparisonActivityStart
    test_bm = benchmark(
        config,
        mode='all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_input_plan_comparison_activity_duration_zero():
    # input plan comparisons self-build benchmark_data_path, do not require separate test classes
    benchmark = benchmarking.InputPlanComparisonActivityDuration
    test_bm = benchmark(
        config,
        mode='all',
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0

def test_duration_comparison_score_zero_filepath():
    benchmark = benchmarking.DurationBreakdownComparison
    benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'trip_duration_breakdown_all.csv')
        )
    test_bm = benchmark(
        config,
        'all',
        benchmark_data_path = benchmark_data_path
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_mode_duration_comparison_score_zero_filepath():
    benchmark = benchmarking.DurationModeBreakdownComparison
    benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'trip_duration_breakdown_mode.csv')
        )
    test_bm = benchmark(
        config,
        'all',
        benchmark_data_path = benchmark_data_path
    )
    score = test_bm.build({}, write_path=test_outputs)
    assert score['mse'] == 0


def test_destination_act_duration_comparison_score_zero_filepath():
    benchmark = benchmarking.DurationDestinationActivityBreakdownComparison
    benchmark_data_path = get_benchmark_data(
            os.path.join('test_fixtures', 'trip_duration_breakdown_d_act.csv')
        )
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

    bm_workstation = benchmarking.BenchmarkWorkStation(test_config)
    bm_workstation.connect(managers=None, suppliers=[event_workstation, plan_workstation])

    bm_workstation.resources['link_counter_comparison:car'] = bm_workstation.tools['link_counter_comparison'](
        config=test_config,
        mode='car',
        benchmark_data_path = get_benchmark_data(
            os.path.join('test_town', 'test_town_cordon', 'test_link_counter.json')
        )
    )

    bm_workstation.build(write_path=test_outputs)


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

    bm_workstation = benchmarking.BenchmarkWorkStation(test_config_dictionary)
    bm_workstation.connect(managers=None, suppliers=[event_workstation, plan_workstation])

    benchmarking_workstation = benchmarking.BenchmarkWorkStation(test_config_dictionary)
    benchmarking_workstation.connect(managers=None, suppliers=[event_workstation, plan_workstation, bm_workstation])
    tool = benchmarking_workstation.tools['duration_breakdown_comparison']
    benchmarking_workstation.resources['duration_breakown_comparison'] = tool(test_config_dictionary, 'all', benchmark_data_path='./tests/test_outputs/trip_duration_breakdown_all.csv')

    benchmarking_workstation.build(write_path=test_outputs)