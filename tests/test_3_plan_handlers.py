import sys
import os
import pytest
import pandas as pd
import numpy as np


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara import plan_handlers
from elara.plan_handlers import PlanHandlerWorkStation
sys.path.append(os.path.abspath('../tests'))


test_matsim_time_data = [
    ('00:00:00', 0),
    ('01:01:01', 3661),
    (None, None),
]


@pytest.mark.parametrize("time,seconds", test_matsim_time_data)
def test_convert_time(time, seconds):
    assert plan_handlers.convert_time(time) == seconds


# Config
@pytest.fixture
def test_config():
    config_path = os.path.join('tests/test_xml_scenario.toml')
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
    base_handler.build(input_manager.resources)
    return base_handler

### Log Handler ###
@pytest.fixture
def agent_log_handler(test_config, input_manager):
    handler = plan_handlers.LogsHandler(test_config, 'all')

    resources = input_manager.resources
    handler.build(resources)

    assert len(handler.activities) == 0
    assert len(handler.legs) == 0

    return handler


def test_agent_log_handler(agent_log_handler):
    handler = agent_log_handler

    plans = handler.resources['plans']
    for plan in plans.elems:
        handler.process_plan(plan)

    assert len(handler.activities) == 23
    assert len(handler.legs) == 18


@pytest.fixture
def agent_log_handler_finalised(agent_log_handler):
    handler = agent_log_handler
    plans = handler.resources['plans']
    for plan in plans.elems:
        handler.process_plan(plan)
    handler.finalise()
    return handler


def test_finalised_mode_counts_car(agent_log_handler_finalised):
    handler = agent_log_handler_finalised

    assert len(handler.results) == 2

    for name, result in handler.results.items():
        if 'activity' in name:
            assert len(result) == 23

        else:
            assert len(result) == 18


### Highway Distance Handler ###
@pytest.fixture
def agent_distances_handler_car_mode(test_config, input_manager):
    handler = plan_handlers.AgentHighwayDistance(test_config, 'car')

    resources = input_manager.resources
    handler.build(resources)

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
    for plan in plans.elems:
        handler.process_plan(plan)

    assert np.sum(handler.distances) == 40600.0

    # agent
    assert np.sum(handler.distances[handler.agent_indices['chris']]) == 20400.0

    # class
    assert np.sum(handler.distances[:, handler.ways_indices['trunk']]) == 30600.0


@pytest.fixture
def agent_distances_handler_finalised_car(agent_distances_handler_car_mode):
    handler = agent_distances_handler_car_mode
    plans = handler.resources['plans']
    for plan in plans.elems:
        handler.process_plan(plan)
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


### Modeshare Handler ###
@pytest.fixture
def test_plan_modeshare_handler(test_config, input_manager):
    handler = plan_handlers.ModeShare(test_config, 'all')

    resources = input_manager.resources
    handler.build(resources)

    periods = 24

    assert len(handler.modes) == len(handler.resources['output_config'].modes)
    assert list(handler.mode_indices.keys()) == handler.modes

    assert len(handler.classes) == len(handler.resources['attribute'].classes)
    assert list(handler.class_indices.keys()) == handler.classes

    assert len(handler.activities) == len(handler.resources['output_config'].activities)
    assert list(handler.activity_indices.keys()) == handler.activities

    assert handler.mode_counts.shape == (
        len(handler.resources['output_config'].modes),
        len(handler.resources['attribute'].classes),
        len(handler.resources['output_config'].activities),
        periods)

    return handler


def test_plan_handler_test_data(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler

    plans = test_plan_modeshare_handler.resources['plans']
    for plan in plans.elems:
        handler.process_plan(plan)

    assert np.sum(handler.mode_counts) == 10

    # mode
    assert np.sum(handler.mode_counts[handler.mode_indices['car']]) == 4
    assert np.sum(handler.mode_counts[handler.mode_indices['bus']]) == 4
    assert np.sum(handler.mode_counts[handler.mode_indices['bike']]) == 2
    assert np.sum(handler.mode_counts[handler.mode_indices['transit_walk']]) == 0

    # class
    assert np.sum(handler.mode_counts[:, handler.class_indices['rich'], :, :]) == 2
    assert np.sum(handler.mode_counts[:, handler.class_indices['poor'], :, :]) == 8
    assert np.sum(handler.mode_counts[:, handler.class_indices['not_applicable'], :, :]) == 0

    # activities
    assert np.sum(handler.mode_counts[:, :, handler.activity_indices['pt interaction'], :]) == 0
    assert np.sum(handler.mode_counts[:, :, handler.activity_indices['work']]) == 5
    assert np.sum(handler.mode_counts[:, :, handler.activity_indices['home']]) == 5

    # time
    assert np.sum(handler.mode_counts[:, :, :, :12]) == 5
    assert np.sum(handler.mode_counts[:, :, :, 12:]) == 5


@pytest.fixture
def test_plan_handler_finalised(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler
    plans = test_plan_modeshare_handler.resources['plans']
    for plan in plans.elems:
        handler.process_plan(plan)
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
    plan_workstation.load_all_tools(option='all')
    plan_workstation.build()

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
