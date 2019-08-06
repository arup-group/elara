import sys
import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import inputs, handlers
from elara.handlers.agent_plan_handlers import *
sys.path.append(os.path.abspath('../tests'))


test_mastim_time_data = [
    ('00:00:00', 0),
    ('01:01:01', 3661),
    (None, None),
]


@pytest.mark.parametrize("time,seconds", test_mastim_time_data)
def test_convert_time(time, seconds):
    assert handlers.agent_plan_handlers.convert_time(time) == seconds


@pytest.fixture
def config():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    return Config(config_path)


@pytest.fixture
def network(config):
    return inputs.Network(config.network_path, config.crs)


@pytest.fixture
def mode_hierarchy():
    return inputs.ModeHierarchy()


@pytest.fixture
def transit_schedule(config):
    return inputs.TransitSchedule(
        config.transit_schedule_path, config.crs
            )


@pytest.fixture
def attributes(config):
    return inputs.Attributes(config.attributes_path)


@pytest.fixture
def transit_vehicles(config):
    return inputs.TransitVehicles(
            config.transit_vehicles_path
            )


@pytest.fixture
def plans(config, transit_schedule):
    return inputs.Plans(config.plans_path, transit_schedule)


# Mode Share
@pytest.fixture
def test_plan_modeshare_handler(
        plans,
        transit_schedule,
        attributes,
        mode_hierarchy,
):
    periods = 24
    handler = ModeShare(
        'all',
        plans,
        transit_schedule,
        attributes,
        mode_hierarchy,
        time_periods=periods,
        scale_factor=0.01
    )
    assert len(handler.modes) == len(plans.modes)
    assert list(handler.mode_indices.keys()) == handler.modes

    assert len(handler.classes) == len(attributes.classes)
    assert list(handler.class_indices.keys()) == handler.classes

    assert len(handler.activities) == len(plans.activities)
    assert list(handler.activity_indices.keys()) == handler.activities

    assert handler.mode_counts.shape == (
        len(plans.modes),
        len(attributes.classes),
        len(plans.activities),
        periods)

    return handler


def test_plan_handler_test_data(test_plan_modeshare_handler, plans):
    handler = test_plan_modeshare_handler
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
def test_plan_handler_finalised(test_plan_modeshare_handler, plans):
    handler = test_plan_modeshare_handler
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
                assert np.sum(df.values) == 10 / handler.scale_factor

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
                assert np.sum(df.values) == 10 / handler.scale_factor


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
