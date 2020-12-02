import sys
import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree
from datetime import datetime, timedelta


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara import plan_handlers
from elara.plan_handlers import PlanHandlerWorkStation

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
test_inputs = os.path.join(test_dir, "test_intermediate_data")
test_outputs = os.path.join(test_dir, "test_outputs")
if not os.path.exists(test_outputs):
    os.mkdir(test_outputs)
benchmarks_path = os.path.join(test_outputs, "benchmarks")
if not os.path.exists(benchmarks_path):
    os.mkdir(benchmarks_path)


test_matsim_time_data = [
    ('00:00:00', 0),
    ('01:01:01', 3661),
    (None, None),
]


@pytest.mark.parametrize("time,seconds", test_matsim_time_data)
def test_convert_time(time, seconds):
    assert plan_handlers.convert_time_to_seconds(time) == seconds


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


# Input Manager
@pytest.fixture
def input_manager(test_config, test_paths):
    input_workstation = InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()
    return input_workstation


# Base
@pytest.fixture
def base_handler(test_config, input_manager):
    base_handler = plan_handlers.PlanHandlerTool(test_config, 'all')
    assert base_handler.option == 'all'
    base_handler.build(input_manager.resources, write_path=test_outputs)
    return base_handler


### Utility Handler ###

@pytest.fixture
def person_single_plan_elem():
    string = """
        <person id="test1">
		<plan score="10" selected="yes">
		</plan>
	</person>
        """
    return etree.fromstring(string)


@pytest.fixture
def person_plans_elem():
    string = """
        <person id="test2">
		<plan score="10" selected="yes">
		</plan>
        <plan score="8" selected="no">
		</plan>
	</person>
        """
    return etree.fromstring(string)


@pytest.fixture
def utility_handler(test_config, input_manager):
    handler = plan_handlers.UtilityHandler(test_config, 'all')
    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    assert len(handler.utility_log.chunk) == 0

    return handler


def test_utility_handler_process_single_plan(utility_handler, person_single_plan_elem):
    assert len(utility_handler.utility_log) == 0
    utility_handler.process_plans(person_single_plan_elem)
    assert len(utility_handler.utility_log) == 1
    assert utility_handler.utility_log.chunk == [{'agent': 'test1','score': '10'}]


def test_utility_handler_process_multi_plan(utility_handler, person_plans_elem):
    assert len(utility_handler.utility_log) == 0
    utility_handler.process_plans(person_plans_elem)
    assert len(utility_handler.utility_log) == 1
    assert utility_handler.utility_log.chunk == [{'agent': 'test2','score': '10'}]


def test_utility_handler_process_plans(utility_handler, person_single_plan_elem, person_plans_elem):
    assert len(utility_handler.utility_log) == 0
    utility_handler.process_plans(person_single_plan_elem)
    utility_handler.process_plans(person_plans_elem)
    assert len(utility_handler.utility_log) == 2
    assert utility_handler.utility_log.chunk == [{'agent': 'test1','score': '10'}, {'agent': 'test2','score': '10'}]


### Log Handler ###
# Wrapping
test_matsim_time_data = [
    (['06:00:00', '12:45:00', '18:30:00'], '1-18:30:00'),
    (['06:00:00', '12:45:00', '24:00:00'], '2-00:00:00'),
    (['06:00:00', '24:00:00', '08:30:00'], '2-08:30:00'),
    (['06:00:00', '18:45:00', '12:30:00'], '2-12:30:00'),
    (['06:00:00', '18:45:00', '18:45:00'], '1-18:45:00'),
    (['00:00:00', '12:45:00', '18:45:00'], '1-18:45:00'),
    (['06:00:00', '04:45:00', '02:45:00'], '3-02:45:00'),
    (['00:00:00'], '1-00:00:00'),
    (['24:00:00'], '2-00:00:00'),
]


# @pytest.mark.parametrize("times,final_string", test_matsim_time_data)
# def test_day_wrapping(times, final_string):
#     current_dt = None
#     for new_time_str in times:
#         current_dt = plan_handlers.non_wrapping_datetime(current_dt, new_time_str)
#     assert isinstance(current_dt, datetime)
#     assert current_dt == datetime.strptime(f"{final_string}", '%d-%H:%M:%S')


non_wrapping_test_matsim_time_data = [
    (['06:00:00', '12:45:00', '18:30:00'], '1-18:30:00'),
    (['06:00:00', '12:45:00', '24:00:00'], '2-00:00:00'),
    (['06:00:00', '24:00:00', '08:30:00'], '1-08:30:00'),
    (['06:00:00', '18:45:00', '12:30:00'], '1-12:30:00'),
    (['00:00:00'], '1-00:00:00'),
    (['24:00:00'], '2-00:00:00'),
]

@pytest.mark.parametrize("times,final_string", non_wrapping_test_matsim_time_data)
def test_matsim_time_to_datetime(times, final_string):
    current_dt = None
    for new_time_str in times:
        current_dt = plan_handlers.matsim_time_to_datetime(
            current_dt,
            new_time_str,
            base_year=1900,
            base_month=1,
            )
    assert isinstance(current_dt, datetime)
    assert current_dt == datetime.strptime(f"{final_string}", '%d-%H:%M:%S')


test_durations_data = [
    (
        None, datetime(year=2020, month=4, day=1, hour=0),
        timedelta(hours=0), datetime(year=2020, month=4, day=1, hour=0)
    ),
    (
        None, datetime(year=2020, month=4, day=1, hour=1),
        timedelta(hours=1), datetime(year=2020, month=4, day=1, hour=0)
    ),
    (
        datetime(year=2020, month=4, day=1, hour=1), datetime(year=2020, month=4, day=1, hour=1),
        timedelta(hours=0), datetime(year=2020, month=4, day=1, hour=1)
    ),
    (
        datetime(year=2020, month=4, day=1, hour=1), datetime(year=2020, month=4, day=1, hour=2),
        timedelta(hours=1), datetime(year=2020, month=4, day=1, hour=1)
    ),
    (
        datetime(year=2020, month=4, day=1, hour=2), datetime(year=2020, month=4, day=1, hour=1),
        timedelta(hours=-1), datetime(year=2020, month=4, day=1, hour=2)
    ),
]
@pytest.mark.parametrize("start,end,duration,start_time", test_durations_data)
def test_safe_duration(start, end, duration, start_time):
    d, st = plan_handlers.safe_duration(start, end)
    assert d == duration
    assert st == start_time


test_distance_data = [
    (0,0,0,0,0),
    (1,1,1,1,0),
    (0,0,3,4,5),
    (3,4,0,0,5),
    (3,0,0,-4,5)
]
@pytest.mark.parametrize("x1,y1,x2,y2,dist", test_distance_data)
def test_distance(x1,y1,x2,y2,dist):
    assert plan_handlers.distance(x1,y1,x2,y2) == dist


# Normal Case
@pytest.fixture
def agent_log_handler(test_config, input_manager):
    handler = plan_handlers.AgentLogsHandler(test_config, 'all')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.activities_log.chunk) == 0
    assert len(handler.legs_log.chunk) == 0

    return handler


def test_agent_log_handler(agent_log_handler):
    handler = agent_log_handler

    plans = handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert len(handler.activities_log.chunk) == 23
    assert len(handler.legs_log.chunk) == 18


@pytest.fixture
def agent_log_handler_finalised(agent_log_handler):
    handler = agent_log_handler
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    handler.finalise()
    return handler


def test_finalised_logs(agent_log_handler_finalised):
    handler = agent_log_handler_finalised

    assert len(handler.results) == 0


# Plans Wrapping case

# Bad Config (plans wrap past 24hrs)
@pytest.fixture
def test_bad_plans_config():
    config_path = os.path.join(test_dir, 'test_xml_scenario_bad_plans.toml')
    config = Config(config_path)
    assert config
    return config


# Paths
@pytest.fixture
def test_bad_plans_paths(test_bad_plans_config):
    paths = PathFinderWorkStation(test_bad_plans_config)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths


# Input Manager
@pytest.fixture
def input_bad_plans_manager(test_bad_plans_config, test_bad_plans_paths):
    input_workstation = InputsWorkStation(test_bad_plans_config)
    input_workstation.connect(managers=None, suppliers=[test_bad_plans_paths])
    input_workstation.load_all_tools()
    input_workstation.build()
    return input_workstation


@pytest.fixture
def agent_log_handler_bad_plans(test_bad_plans_config, input_bad_plans_manager):
    handler = plan_handlers.AgentLogsHandler(test_bad_plans_config, 'all')

    resources = input_bad_plans_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.activities_log.chunk) == 0
    assert len(handler.legs_log.chunk) == 0

    return handler


@pytest.fixture
def agent_log_handler_finalised_bad_plans(agent_log_handler_bad_plans):
    handler = agent_log_handler_bad_plans
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    assert handler.activities_log.chunk[-1].get('end_day') == 1
    assert handler.legs_log.chunk[-1].get('end_day') == 1
    handler.finalise()
    return handler


def test_finalised_logs_bad_plans(agent_log_handler_finalised_bad_plans):
    handler = agent_log_handler_finalised_bad_plans
    assert len(handler.results) == 0


# Plan Handler ###
@pytest.fixture
def agent_plan_handler(test_bad_plans_config, input_manager):
    handler = plan_handlers.AgentPlansHandler(test_bad_plans_config, 'poor')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    assert len(handler.plans_log.chunk) == 0
    return handler


def test_agent_plans_handler(agent_plan_handler):
    handler = agent_plan_handler

    plans = handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert len(handler.plans_log.chunk) == 8


@pytest.fixture
def agent_plans_handler_finalised(agent_plan_handler):
    handler = agent_plan_handler
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    handler.finalise()
    return handler


def test_finalised_plans(agent_plans_handler_finalised):
    handler = agent_plans_handler_finalised

    assert len(handler.results) == 0


# Plans Wrapping case

# Bad Config (plans wrap past 24hrs)
@pytest.fixture
def agent_plans_handler_bad_plans(test_bad_plans_config, input_bad_plans_manager):
    handler = plan_handlers.AgentPlansHandler(test_bad_plans_config, 'poor')

    resources = input_bad_plans_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.plans_log.chunk) == 0
    return handler


# @pytest.fixture
# def agent_plans_handler_finalised_bad_plans(agent_plans_handler_bad_plans):
#     handler = agent_plans_handler_bad_plans
#     plans = handler.resources['plans']
#     for plan in plans.persons:
#         handler.process_plans(plan)
#     assert handler.plans_log.chunk[-1].get('end_day') == 1
#     handler.finalise()
#     return handler


# def test_finalised_plans_bad_plans(agent_plans_handler_finalised_bad_plans):
#     handler = agent_plans_handler_finalised_bad_plans
#     assert len(handler.results) == 0


### Agent Highway Distance Handler ###
@pytest.fixture
def agent_distances_handler_car_mode(test_config, input_manager):
    handler = plan_handlers.AgentHighwayDistanceHandler(test_config, 'car')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.agents_ids) == len(handler.resources['agents'].idents)
    assert list(handler.agent_indices.keys()) == handler.agents_ids

    assert len(handler.ways) == len(handler.resources['osm:ways'].classes)
    assert list(handler.ways_indices.keys()) == handler.ways

    assert handler.distances.shape == (
        len(handler.resources['agents'].idents),
        len(handler.resources['osm:ways'].classes)
    )

    return handler


def test_agent_distances_handler_car_mode(agent_distances_handler_car_mode):
    handler = agent_distances_handler_car_mode

    plans = handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert np.sum(handler.distances) == 40600.0

    # agent
    assert np.sum(handler.distances[handler.agent_indices['chris']]) == 20400.0

    # class
    assert np.sum(handler.distances[:, handler.ways_indices['trunk']]) == 30600.0


@pytest.fixture
def agent_distances_handler_finalised_car(agent_distances_handler_car_mode):
    handler = agent_distances_handler_car_mode
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    handler.finalise()
    return handler


def test_finalised_agent_distances_car(agent_distances_handler_finalised_car):
    handler = agent_distances_handler_finalised_car

    for name, result in handler.results.items():
        cols = handler.ways
        if 'total' in name:
            for c in cols:
                assert c in result.index
            assert 'total' in result.index
            assert np.sum(result[cols].values) == 40600.0

        else:
            for c in cols:
                assert c in result.columns
            assert 'total' in result.columns
            df = result.loc[:, cols]
            assert np.sum(df.values) == 40600.0


### Trips Highway Distance Handler ###
@pytest.fixture
def trip_distances_handler_car_mode(test_config, input_manager):
    handler = plan_handlers.TripHighwayDistanceHandler(test_config, 'car')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.ways) == len(handler.resources['osm:ways'].classes)

    return handler


def test_trip_distances_handler_car_mode(trip_distances_handler_car_mode):
    handler = trip_distances_handler_car_mode

    plans = handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert len(handler.distances_log.chunk) == 10
    assert sum([d['None'] for d in handler.distances_log.chunk]) == 10000
    assert sum([d['trunk'] for d in handler.distances_log.chunk]) == 30600

    # agent
    assert sum([d['trunk'] for d in handler.distances_log.chunk if d['agent'] == 'chris']) == 20400.0


def test_trip_distances_handler_finalised_car(trip_distances_handler_car_mode):
    handler = trip_distances_handler_car_mode
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    handler.finalise()

    path = handler.distances_log.path
    results = pd.read_csv(path)
    assert len(results) == 10
    assert sum(results.trunk) == 30600
    assert sum(results.loc[results.agent == 'chris'].trunk) == 20400


### Modeshare Handler ###
@pytest.fixture
def test_plan_modeshare_handler(test_config, input_manager):
    handler = plan_handlers.ModeShareHandler(test_config, 'all')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert len(handler.modes) == len(handler.resources['output_config'].modes)
    assert list(handler.mode_indices.keys()) == handler.modes

    assert len(handler.classes) == len(handler.resources['attributes'].classes)
    assert list(handler.class_indices.keys()) == handler.classes

    assert len(handler.activities) == len(handler.resources['output_config'].activities)
    assert list(handler.activity_indices.keys()) == handler.activities

    assert handler.mode_counts.shape == (
        len(handler.resources['output_config'].modes),
        len(handler.resources['attributes'].classes),
        len(handler.resources['output_config'].activities),
        periods)

    return handler


def test_plan_handler_test_data(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler

    plans = test_plan_modeshare_handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert np.sum(handler.mode_counts) == 10

    # mode
    assert np.sum(handler.mode_counts[handler.mode_indices['car']]) == 4
    assert np.sum(handler.mode_counts[handler.mode_indices['bus']]) == 4
    assert np.sum(handler.mode_counts[handler.mode_indices['bike']]) == 2
    assert np.sum(handler.mode_counts[handler.mode_indices['walk']]) == 0

    # class
    assert np.sum(handler.mode_counts[:, handler.class_indices['rich'], :, :]) == 2
    assert np.sum(handler.mode_counts[:, handler.class_indices['poor'], :, :]) == 8
    # assert np.sum(handler.mode_counts[:, handler.class_indices['not_applicable'], :, :]) == 0

    # activities
    # assert np.sum(handler.mode_counts[:, :, handler.activity_indices['pt interaction'], :]) == 0
    assert np.sum(handler.mode_counts[:, :, handler.activity_indices['work']]) == 5
    assert np.sum(handler.mode_counts[:, :, handler.activity_indices['home']]) == 5

    # time
    assert np.sum(handler.mode_counts[:, :, :, :12]) == 5
    assert np.sum(handler.mode_counts[:, :, :, 12:]) == 5


@pytest.fixture
def test_plan_handler_finalised(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler
    plans = test_plan_modeshare_handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)
    handler.finalise()
    return handler


def test_finalised_mode_counts(test_plan_handler_finalised):
    handler = test_plan_handler_finalised

    for name, result in handler.results.items():
        if 'count' in name:
            cols = handler.modes
            if isinstance(result, pd.DataFrame):
                for c in cols:
                    assert c in result.columns
                df = result.loc[:, cols]
                assert np.sum(df.values) == 10 / handler.config.scale_factor

                if 'class' in result.columns:
                    assert set(result.loc[:, 'class']) == set(handler.classes)
                if 'activity' in result.columns:
                    assert set(result.loc[:, 'activity']) == set(handler.activities)
                if 'hour' in result.columns:
                    assert set(result.loc[:, 'hour']) == set(range(24))
            else:
                for c in cols:
                    assert c in result.index
                df = result.loc[cols]
                assert np.sum(df.values) == 10 / handler.config.scale_factor


def test_finalised_mode_shares(test_plan_handler_finalised):
    handler = test_plan_handler_finalised

    for name, result in handler.results.items():
        if 'share' in name:
            cols = handler.modes
            if isinstance(result, pd.DataFrame):
                for c in cols:
                    assert c in result.columns
                df = result.loc[:, cols]
                assert np.sum(df.values) == 1

                if 'class' in result.columns:
                    assert set(result.loc[:, 'class']) == set(handler.classes)
                if 'activity' in result.columns:
                    assert set(result.loc[:, 'activity']) == set(handler.activities)
                if 'hour' in result.columns:
                    assert set(result.loc[:, 'hour']) == set(range(24))
            else:
                for c in cols:
                    assert c in result.index
                df = result.loc[cols]
                assert np.sum(df.values) == 1


# Plan Handler Manager
def test_load_plan_handler_manager(test_config, test_paths):
    input_workstation = InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    plan_workstation = PlanHandlerWorkStation(test_config)
    plan_workstation.connect(managers=None, suppliers=[input_workstation])

    tool = plan_workstation.tools['mode_share']
    plan_workstation.resources['mode_share'] = tool(test_config, 'all')

    plan_workstation.build(write_path=test_outputs)

    for handler in plan_workstation.resources.values():
        for name, result in handler.results.items():
            if 'share' in name:
                cols = handler.modes
                if isinstance(result, pd.DataFrame):
                    for c in cols:
                        assert c in result.columns
                    df = result.loc[:, cols]
                    assert np.sum(df.values) == 1

                    if 'class' in result.columns:
                        assert set(result.loc[:, 'class']) == set(handler.classes)
                    if 'activity' in result.columns:
                        assert set(result.loc[:, 'activity']) == set(handler.activities)
                    if 'hour' in result.columns:
                        assert set(result.loc[:, 'hour']) == set(range(24))
                else:
                    for c in cols:
                        assert c in result.index
                    df = result.loc[cols]
                    assert np.sum(df.values) == 1
