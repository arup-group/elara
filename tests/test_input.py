import sys
import os

sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import inputs
sys.path.append(os.path.abspath('../tests'))


# Config
def test_config_of_xml_inputs():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    assert config


def test_config_of_gzip_inputs():
    config_path = os.path.join('tests/test_gzip_scenario.toml')
    config = Config(config_path)
    assert config


# Events
def test_loading_xml_events():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    events = inputs.Events(config.events_path).elems
    event = next(events)
    assert event is not None


def test_loading_gzip_events():
    config_path = os.path.join('tests/test_gzip_scenario.toml')
    config = Config(config_path)
    events = inputs.Events(config.events_path).elems
    event = next(events)
    assert event is not None


# Plans
def test_loading_xml_plans():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    transit_schedule = inputs.TransitSchedule(
        config.transit_schedule_path, config.crs
    )
    assert len(transit_schedule.stop_gdf)
    plans = inputs.Plans(config.plans_path, transit_schedule).elems
    plan = next(plans)
    assert plan is not None


def test_loading_gzip_plans():
    config_path = os.path.join('tests/test_gzip_scenario.toml')
    config = Config(config_path)
    transit_schedule = inputs.TransitSchedule(
        config.transit_schedule_path, config.crs
    )
    assert len(transit_schedule.stop_gdf)
    plans = inputs.Plans(config.plans_path, transit_schedule).elems
    plan = next(plans)
    assert plan is not None


# Network
def test_loading_xml_network():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    network  = inputs.Network(config.network_path, config.crs)
    assert len(network.link_gdf)


def test_loading_gzip_network():
    config_path = os.path.join('tests/test_gzip_scenario.toml')
    config = Config(config_path)
    network  = inputs.Network(config.network_path, config.crs)
    assert len(network.link_gdf)

# TODO add other inputs
