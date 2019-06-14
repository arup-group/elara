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


config_path = os.path.join('tests/test_xml_scenario.toml')
xml_config = Config(config_path)

config_path = os.path.join('tests/test_gzip_scenario.toml')
gzip_config = Config(config_path)


# Events
def test_loading_xml_events():
    events = inputs.Events(xml_config.events_path).elems
    num_events = sum(1 for _ in events)
    assert num_events == 190


def test_loading_gzip_events():
    events = inputs.Events(gzip_config.events_path).elems
    num_events = sum(1 for _ in events)
    assert num_events == 190


# Transit
def test_load_xml_transit_schedule():
    transit_schedule = inputs.TransitSchedule(
            xml_config.transit_schedule_path, xml_config.crs
            )
    assert len(transit_schedule.stop_gdf) == 4


def test_load_gzip_transit_schedule():
    transit_schedule = inputs.TransitSchedule(
            gzip_config.transit_schedule_path, gzip_config.crs
            )
    assert len(transit_schedule.stop_gdf) == 4


# Plans
def test_loading_xml_plans():
    transit_schedule = inputs.TransitSchedule(
        xml_config.transit_schedule_path, xml_config.crs
    )
    assert len(transit_schedule.stop_gdf)
    plans = inputs.Plans(xml_config.plans_path, transit_schedule).elems
    num_plans = sum(1 for _ in plans)
    assert num_plans == 5


def test_loading_gzip_plans():
    transit_schedule = inputs.TransitSchedule(
        gzip_config.transit_schedule_path, gzip_config.crs
    )
    assert len(transit_schedule.stop_gdf)
    plans = inputs.Plans(gzip_config.plans_path, transit_schedule).elems
    num_plans = sum(1 for _ in plans)
    assert num_plans == 5


# Network
def test_loading_xml_network():
    network  = inputs.Network(xml_config.network_path, xml_config.crs)
    assert len(network.link_gdf) == 8


def test_loading_gzip_network():
    network  = inputs.Network(gzip_config.network_path, gzip_config.crs)
    assert len(network.link_gdf) == 8
