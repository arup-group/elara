import sys
import os
import pytest


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, PathFinderWorkStation
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


@pytest.fixture
def test_paths(test_xml_config):
    paths = PathFinderWorkStation(test_xml_config)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths


@pytest.fixture
def test_zip_paths(test_gzip_config):
    paths = PathFinderWorkStation(test_gzip_config)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths


# Events
def test_loading_xml_events(test_xml_config, test_paths):
    events = inputs.Events(test_xml_config)
    events.build(test_paths.resources)
    num_events = sum(1 for _ in events.elems)
    assert num_events == 190


def test_loading_gzip_events(test_gzip_config, test_zip_paths):
    events = inputs.Events(test_gzip_config)
    events.build(test_zip_paths.resources)
    num_events = sum(1 for _ in events.elems)
    assert num_events == 190


# Modemap
def test_modemap_get(test_xml_config, test_paths):
    modemap = inputs.ModeMap(test_xml_config)
    modemap.build(test_paths.resources)
    assert modemap['egress_walk'] == 'walk'


# Transit
def test_load_xml_transit_schedule(test_xml_config, test_paths):
    transit_schedule = inputs.TransitSchedule(test_xml_config)
    transit_schedule.build(test_paths.resources)
    assert len(transit_schedule.stop_gdf) == 4


def test_load_gzip_transit_schedule(test_gzip_config, test_zip_paths):
    transit_schedule = inputs.TransitSchedule(test_gzip_config)
    transit_schedule.build(test_zip_paths.resources)
    assert len(transit_schedule.stop_gdf) == 4


def test_load_xml_transit_vehicles(test_xml_config, test_paths):
    transit_vehicles = inputs.TransitVehicles(test_xml_config)
    transit_vehicles.build(test_paths.resources)
    assert transit_vehicles.veh_type_capacity_map['Bus'] == 70
    assert len(
        transit_vehicles.veh_id_veh_type_map
    ) == sum(
        transit_vehicles.transit_vehicle_counts.values()
    )


def test_load_gzip_transit_vehicles(test_gzip_config, test_zip_paths):
    transit_vehicles = inputs.TransitVehicles(test_gzip_config)
    transit_vehicles.build(test_zip_paths.resources)
    assert transit_vehicles.veh_type_capacity_map['Bus'] == 70
    assert len(
        transit_vehicles.veh_id_veh_type_map
    ) == sum(
        transit_vehicles.transit_vehicle_counts.values()
    )


# Plans
def test_loading_xml_plans(test_xml_config, test_paths):
    plans = inputs.Plans(test_xml_config)
    plans.build(test_paths.resources)
    num_plans = sum(1 for _ in plans.elems)
    assert num_plans == 5


def test_loading_gzip_plans(test_gzip_config, test_zip_paths):
    plans = inputs.Plans(test_gzip_config)
    plans.build(test_zip_paths.resources)
    num_plans = sum(1 for _ in plans.elems)
    assert num_plans == 5


# ModeHierarchy
def test_hierarchy_get_bad_type():
    hierarchy = inputs.ModeHierarchy(None)
    with pytest.raises(TypeError):
        assert hierarchy['egress_walk']
    with pytest.raises(TypeError):
        assert hierarchy[[1]]


def test_hierarchy_get_unknown(capsys):
    hierarchy = inputs.ModeHierarchy(None)
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
    hierarchy = inputs.ModeHierarchy(None)
    assert hierarchy.get(modes) == mode


# Network
def test_loading_xml_network(test_xml_config, test_paths):
    network = inputs.Network(test_xml_config)
    network.build(test_paths.resources)
    assert len(network.link_gdf) == 8
    assert len(network.node_gdf) == 5


def test_loading_gzip_network(test_gzip_config, test_zip_paths):
    network = inputs.Network(test_gzip_config)
    network.build(test_zip_paths.resources)
    assert len(network.link_gdf) == 8
    assert len(network.node_gdf) == 5


# OSMWay
def test_loading_osm_highways_map_from_xml(test_xml_config, test_paths):
    osm_highway = inputs.OSMWays(test_xml_config)
    osm_highway.build(test_paths.resources)
    assert len(osm_highway.ways) == 8
    assert len(osm_highway.classes) == 2


def test_loading_osm_highways_map_from_gzip(test_gzip_config, test_zip_paths):
    osm_highway = inputs.OSMWays(test_gzip_config)
    osm_highway.build(test_zip_paths.resources)
    assert len(osm_highway.ways) == 8
    assert len(osm_highway.classes) == 2


# Attribute
def test_loading_xml_attribute(test_xml_config, test_paths):
    attribute = inputs.Attributes(test_xml_config)
    attribute.build(test_paths.resources)
    assert len(attribute.map) == sum(attribute.attribute_count_map.values())


def test_loading_gzip_attribute(test_gzip_config, test_zip_paths):
    attribute = inputs.Attributes(test_gzip_config)
    attribute.build(test_zip_paths.resources)
    assert len(attribute.map) == sum(attribute.attribute_count_map.values())


# Agents
def test_loading_xml_attributes(test_xml_config, test_paths):
    attributes = inputs.Agents(test_xml_config)
    attributes.build(test_paths.resources)
    assert len(attributes.map) == 5
    assert attributes.idents == ['chris', 'fatema', 'fred', 'gerry', 'nick']
    assert attributes.attribute_fields == {'age', 'subpopulation'}
    assert len(attributes.attributes_df) == 5


def test_loading_gzip_attributes(test_gzip_config, test_zip_paths):
    attributes = inputs.Agents(test_gzip_config)
    attributes.build(test_zip_paths.resources)
    assert len(attributes.map) == 5
    assert attributes.idents == ['chris', 'fatema', 'fred', 'gerry', 'nick']
    assert attributes.attribute_fields == {'age', 'subpopulation'}
    assert len(attributes.attributes_df) == 5


# Output Config
def test_load_xml_output_config(test_xml_config, test_paths):
    out_config = inputs.OutputConfig(test_xml_config)
    out_config.build(test_paths.resources)
    assert set(out_config.activities) == set(['home', 'work'])
    assert set(out_config.modes) == set(['pt', 'walk', 'bike', 'car', 'transit_walk'])
    assert set(out_config.sub_populations) == set(['rich', 'poor', 'default'])


# Input Manager
def test_load_input_manager(test_xml_config, test_paths):
    input_workstation = inputs.InputsWorkStation(test_xml_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    events = input_workstation.resources['events']
    network = input_workstation.resources['network']
    transit_schedule = input_workstation.resources['transit_schedule']
    transit_vehicles = input_workstation.resources['transit_vehicles']
    attributes = input_workstation.resources['attributes']
    plans = input_workstation.resources['plans']
    mode_map = input_workstation.resources['mode_map']
    mode_hierarchy = input_workstation.resources['mode_hierarchy']

    num_events = sum(1 for _ in events.elems)
    assert num_events == 190

    assert len(network.link_gdf) == 8
    assert len(network.node_gdf) == 5

    assert len(transit_schedule.stop_gdf) == 4

    assert transit_vehicles.veh_type_capacity_map['Bus'] == 70
    assert len(
        transit_vehicles.veh_id_veh_type_map
    ) == sum(
        transit_vehicles.transit_vehicle_counts.values()
    )

    assert len(attributes.map) == sum(attributes.attribute_count_map.values())

    num_plans = sum(1 for _ in plans.elems)
    assert num_plans == 5

    assert mode_map
    assert mode_hierarchy
