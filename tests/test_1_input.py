import sys
import os
import pytest


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import inputs
sys.path.append(os.path.abspath('../tests'))


# Config
@pytest.fixture
def test_xml_config():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    assert config
    return config


@pytest.fixture
def test_gzip_config():
    config_path = os.path.join('tests/test_gzip_scenario.toml')
    config = Config(config_path)
    assert config
    return config


# Events
def test_loading_xml_events(test_xml_config):
    events = inputs.Events(test_xml_config.events_path).elems
    num_events = sum(1 for _ in events)
    assert num_events == 190


def test_loading_gzip_events(test_gzip_config):
    events = inputs.Events(test_gzip_config.events_path).elems
    num_events = sum(1 for _ in events)
    assert num_events == 190


# Modemap
def test_modemap_get():
    modemap = inputs.ModeMap()
    assert modemap['egress_walk'] == 'walk'


# Transit
def test_load_xml_transit_schedule(test_xml_config):
    transit_schedule = inputs.TransitSchedule(
        test_xml_config.transit_schedule_path, test_xml_config.crs
            )
    assert len(transit_schedule.stop_gdf) == 4


def test_load_gzip_transit_schedule(test_gzip_config):
    transit_schedule = inputs.TransitSchedule(
        test_gzip_config.transit_schedule_path, test_gzip_config.crs
            )
    assert len(transit_schedule.stop_gdf) == 4


def test_load_xml_transit_vehicles(test_xml_config):
    transit_vehicles = inputs.TransitVehicles(
        test_xml_config.transit_vehicles_path
            )
    assert transit_vehicles.veh_type_capacity_map['Bus'] == 70
    assert len(
        transit_vehicles.veh_id_veh_type_map
    ) == sum(
        transit_vehicles.transit_vehicle_counts.values()
    )


def test_load_gzip_transit_vehicles(test_gzip_config):
    transit_vehicles = inputs.TransitVehicles(
        test_gzip_config.transit_vehicles_path
            )
    assert transit_vehicles.veh_type_capacity_map['Bus'] == 70
    assert len(
        transit_vehicles.veh_id_veh_type_map
    ) == sum(
        transit_vehicles.transit_vehicle_counts.values()
    )


# Plans
def test_loading_xml_plans(test_xml_config):
    transit_schedule = inputs.TransitSchedule(
        test_xml_config.transit_schedule_path, test_xml_config.crs
    )
    assert len(transit_schedule.stop_gdf)
    plans = inputs.Plans(test_xml_config.plans_path, transit_schedule).elems
    num_plans = sum(1 for _ in plans)
    assert num_plans == 5


def test_loading_gzip_plans(test_gzip_config):
    transit_schedule = inputs.TransitSchedule(
        test_gzip_config.transit_schedule_path, test_gzip_config.crs
    )
    assert len(transit_schedule.stop_gdf)
    plans = inputs.Plans(test_gzip_config.plans_path, transit_schedule).elems
    num_plans = sum(1 for _ in plans)
    assert num_plans == 5


# ModeHierarchy
def test_hierarchy_get_bad_type():
    hierarchy = inputs.ModeHierarchy()
    with pytest.raises(TypeError):
        assert hierarchy['egress_walk']
    with pytest.raises(TypeError):
        assert hierarchy[[1]]


def test_hierarchy_get_unknown(capsys):
    hierarchy = inputs.ModeHierarchy()
    modes = ['one', 'two', 'three']
    mode = hierarchy.get(modes)
    captured = capsys.readouterr()
    assert captured.out
    assert mode == modes[1]


test_hierarchy_get_data = [
    (['transit_walk', 'bus', 'egress_walk'], "bus"),
    (['transit_walk', 'bus', 'rail', 'transit_walk'], "rail"),
    (['car', 'helicopter', 'transit_walk'], "car"),
]


@pytest.mark.parametrize("modes,mode", test_hierarchy_get_data)
def test_hierarchy_get_(modes, mode):
    hierarchy = inputs.ModeHierarchy()
    assert hierarchy.get(modes) == mode


# Network
def test_loading_xml_network(test_xml_config):
    network = inputs.Network(test_xml_config.network_path, test_xml_config.crs)
    assert len(network.link_gdf) == 8
    assert len(network.node_gdf) == 5


def test_loading_gzip_network(test_gzip_config):
    network = inputs.Network(test_gzip_config.network_path, test_gzip_config.crs)
    assert len(network.link_gdf) == 8
    assert len(network.node_gdf) == 5


# Attributes
def test_loading_xml_attributes(test_xml_config):
    attributes = inputs.Attributes(test_xml_config.attributes_path)
    assert len(attributes.map) == sum(attributes.attribute_count_map.values())


def test_loading_gzip_attributes(test_gzip_config):
    attributes = inputs.Attributes(test_gzip_config.attributes_path)
    assert len(attributes.map) == sum(attributes.attribute_count_map.values())

