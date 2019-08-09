import sys
import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree


sys.path.append(os.path.abspath('../elara'))
from elara import factory
from elara.config import Config
from elara import inputs
from elara.factory import *
sys.path.append(os.path.abspath('../tests'))


test_handlers = [
    'volume_counts',
    'passenger_counts',
    'stop_interactions',
    'mode_share',
]

test_inputs = [
        'events',
        'plans',
        'network',
        'attributes',
        'transit_schedule',
        'transit_vehicles',
        'mode_hierarchy',
        'mode_map'
    ]

test_paths = [
        'crs',
        'events',
        'plans',
        'network',
        'attributes',
        'transit_schedule',
        'transit_vehicles',
        'mode_hierarchy',
        'mode_map'
    ]


def test_no_priorities():
    assert [a for a in ConfigManager._prioritise(ConfigManager, [1, 2, 3])] == [1, 2, 3]


@pytest.fixture
def config_path():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    assert os.path.exists(config_path)
    return config_path


@pytest.fixture
def bad_config_path():
    config_path = os.path.join('tests/test_bad_xml_scenario.toml')
    assert os.path.exists(config_path)
    return config_path


def test_handler_pipe_build(config_path):
    supervisor = factory.Supervisor(config_path)
    handler_manager = supervisor.supplier
    assert isinstance(handler_manager, HandlerManager)
    input_manager = handler_manager.supplier
    assert isinstance(input_manager, InputManager)
    config_manager = input_manager.supplier
    assert isinstance(config_manager, ConfigManager)


def test_handler_bad_pipe_build(bad_config_path):
    with pytest.raises(ConfigError):
        supervisor = factory.Supervisor(bad_config_path)


@pytest.fixture
def config(config_path):
    return ConfigManager(config_path)


@pytest.fixture
def test_supervisor(config_path):
    return factory.Supervisor(config_path)


@pytest.fixture
def test_handler_manager(test_supervisor):
    return HandlerManager(test_supervisor)


@pytest.fixture
def test_input_manager(test_handler_manager):
    return InputManager(test_handler_manager)


def test_handler_manager_handler_requirements(test_handler_manager):
    resources = test_handler_manager.manager.requirements()
    assert set(resources) == set(test_handlers)


def test_handler_manager_get_requirements(test_handler_manager):
    resources = test_handler_manager.requirements()
    assert set(resources) == set(test_inputs)


def test_input_manager_input_requirements(test_input_manager):
    resources = test_input_manager.manager.requirements()
    assert set(resources) == set(test_inputs)


def test_input_manager_get_requirements(test_input_manager):
    resources = test_input_manager.requirements()
    assert set(resources) == set(test_paths)






