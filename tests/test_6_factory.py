import sys
import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import inputs
from elara import factory
from elara.handlers import HandlerManager
from elara.handlers.network_event_handlers import *
sys.path.append(os.path.abspath('../tests'))


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


@pytest.fixture
def config_path():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    assert os.path.exists(config_path)
    return config_path


@pytest.fixture
def config():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    return Config(config_path)


@pytest.fixture
def test_supervisor(config_path):
    return factory.Supervisor(config_path)


@pytest.fixture
def test_handler_manager(config):
    return HandlerManager(config)


def test_handler_manager_print_requirements(test_handler_manager):
    test_handler_manager.print_requirements()


def test_handler_manager_get_requirements(test_handler_manager):
    feeds, resources = test_handler_manager.requirements
    assert set(feeds) == {
        'events',
        'plans',
    }

    assert set(resources) == {
        'plans',
        'network',
        'transit_schedule',
        'transit_vehicles',
        'attributes',
        'mode_hierarchy'
    }


def test_config_read_events_path(config):
    assert config.events == getattr(config, "events")
    for name in test_inputs:
        getattr(config, name)


def test_config_validate_config_paths(config):
    config.validate_required_paths(test_inputs)


def test_supervisor_validate_config(test_supervisor):
    test_supervisor.validate_config()

    assert set(test_supervisor.feed_list) == {
        'events',
        'plans',
    }

    assert set(test_supervisor.resource_list) == {
        'plans',
        'network',
        'transit_schedule',
        'transit_vehicles',
        'attributes',
        'mode_hierarchy'
    }


@pytest.fixture
def prepared_supervisor(config_path):
    prepared_supervisor = factory.Supervisor(config_path)
    prepared_supervisor.validate_config()
    prepared_supervisor.prepare_feeds()
    prepared_supervisor.prepare_resources()
    return prepared_supervisor


def test_supervisor_prepare_feeds(prepared_supervisor):
    assert isinstance(prepared_supervisor.live_feeds['events'], inputs.Events)


def test_supervisor_prepare_resources(prepared_supervisor):
    assert isinstance(prepared_supervisor.live_resources['network'], inputs.Network)
    assert len(prepared_supervisor.live_resources['network'].node_gdf) == 5


def test_supervisor_prepare_event_handlers(prepared_supervisor):
    print(prepared_supervisor.config.event_handlers)
    prepared_supervisor.prepare_handlers()




