import sys
import os
import pytest
from shapely.geometry import Point
import geopandas as gpd

# paths in config files etc. assume we're in the repo's root, so make sure we always are
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root_dir)

from elara.config import Config, PathFinderWorkStation
from elara import inputs

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
test_inputs = os.path.join(test_dir, "test_intermediate_data")
test_outputs = os.path.join(test_dir, "test_outputs")
if not os.path.exists(test_outputs):
    os.mkdir(test_outputs)
benchmarks_path = os.path.join(test_outputs, "benchmarks")
if not os.path.exists(benchmarks_path):
    os.mkdir(benchmarks_path)


# Config
@pytest.fixture
def test_xml_config():
    config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
    config = Config(config_path)
    assert config
    return config


@pytest.fixture
def test_xml_config_v12():
    config_path = os.path.join(test_dir, 'test_xml_scenario_v12.toml')
    config = Config(config_path)
    assert config
    return config


@pytest.fixture
def test_gzip_config():
    config_path = os.path.join(test_dir, 'test_gzip_scenario.toml')
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
def test_paths_v12(test_xml_config_v12):
    paths = PathFinderWorkStation(test_xml_config_v12)
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


# Re-project method
@pytest.fixture
def example_gdf():
    data = {
        'a': [1, 2],
        'geometry': [Point(0, 0), Point(1000, 1000)]
    }
    return gpd.GeoDataFrame(data, geometry='geometry')


test_hierarchy_get_data = [
    ('epsg:27700', 'epsg:4326', 'epsg:4326'),
    (None, 'epsg:4326', None),
    ('epsg:27700', None, 'epsg:27700')
]


@pytest.mark.parametrize("set_crs,to_crs,result_crs", test_hierarchy_get_data)
def test_set_reproject(example_gdf, set_crs, to_crs, result_crs):
    input_example = inputs.InputTool(None)
    input_example.set_and_change_crs(target=example_gdf, set_crs=set_crs, to_crs=to_crs)
    assert example_gdf.crs == result_crs


# Events
def test_instantiated_class_names(test_xml_config):
    events = inputs.Events(test_xml_config)
    assert str(events) == "Events"


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


def test_builds_route_to_mode_lookup_from_gzip_transit_schedule(test_gzip_config, test_zip_paths):
    transit_schedule = inputs.TransitSchedule(test_gzip_config)
    transit_schedule.build(test_zip_paths.resources)
    assert transit_schedule.route_to_mode_map == {
        'work_bound': 'bus',
        'home_bound': 'bus'
    }


def test_builds_mode_to_routes_lookup_from_gzip_transit_schedule(test_gzip_config, test_zip_paths):
    transit_schedule = inputs.TransitSchedule(test_gzip_config)
    transit_schedule.build(test_zip_paths.resources)
    assert transit_schedule.mode_to_routes_map == {
        'bus': {'work_bound', 'home_bound'}
    }


def test_builds_modes_to_stops_lookup_from_gzip_transit_schedule(test_gzip_config, test_zip_paths):
    transit_schedule = inputs.TransitSchedule(test_gzip_config)
    transit_schedule.build(test_zip_paths.resources)
    assert transit_schedule.mode_to_stops_map == {
        'bus': {'work_stop_in', 'home_stop_in', 'work_stop_out', 'home_stop_out'}
    }


def test_builds_vehicle_to_route_lookup_from_gzip_transit_schedule(test_gzip_config, test_zip_paths):
    transit_schedule = inputs.TransitSchedule(test_gzip_config)
    transit_schedule.build(test_zip_paths.resources)
    assert transit_schedule.veh_to_route_map == {
        'bus1': 'work_bound',
        'bus2': 'work_bound',
        'bus3': 'home_bound',
        'bus4': 'home_bound'
    }

    
def test_builds_mode_to_vehicle_lookup_from_gzip_transit_schedule(test_gzip_config, test_zip_paths):
    transit_schedule = inputs.TransitSchedule(test_gzip_config)
    transit_schedule.build(test_zip_paths.resources)
    assert transit_schedule.mode_to_veh_map == {
        'bus': {'bus1', 'bus2', 'bus3', 'bus4'}
    }


def test_builds_vehicle_to_mode_lookup_from_gzip_transit_schedule(test_gzip_config, test_zip_paths):
    transit_schedule = inputs.TransitSchedule(test_gzip_config)
    transit_schedule.build(test_zip_paths.resources)
    assert transit_schedule.veh_to_mode_map == {
        'bus1': 'bus',
        'bus2': 'bus',
        'bus3': 'bus',
        'bus4': 'bus'
    }


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
    num_plans = sum(1 for person in plans.persons for plan in person.xpath(".//plan"))
    assert num_plans == 5


def test_loading_gzip_plans(test_gzip_config, test_zip_paths):
    plans = inputs.Plans(test_gzip_config)
    plans.build(test_zip_paths.resources)
    num_plans = sum(1 for person in plans.persons for plan in person.xpath(".//plan"))
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
    assert captured
    assert mode == modes[0]


test_hierarchy_get_data = [
    (['transit_walk', 'bus', 'egress_walk'], "bus"),
    (['transit_walk', 'bus', 'rail', 'transit_walk'], "rail"),
    (['car', 'helicopter', 'transit_walk'], "helicopter"),
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

    
def test_loading_gzip_network(test_gzip_config, test_zip_paths):
    network = inputs.Network(test_gzip_config)
    network.build(test_zip_paths.resources)
    assert network.mode_to_links_map == {'bus': {'2-3', '1-2', '3-4', '2-1', '5-1', '3-2', '1-5', '4-3'}, 'car': {'2-3', '1-2', '3-4', '2-1', '5-1', '3-2', '1-5', '4-3'}}


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


# Subpopulations
def test_loading_xml_subpops(test_xml_config, test_paths):
    attribute = inputs.Subpopulations(test_xml_config)
    attribute.build(test_paths.resources)
    assert attribute.map == {'chris': 'rich', 'fatema': 'poor', 'gerry': 'poor', 'fred': 'poor', 'nick': 'poor'}


def test_loading_gzip_subpops(test_gzip_config, test_zip_paths):
    attribute = inputs.Subpopulations(test_gzip_config)
    attribute.build(test_zip_paths.resources)
    assert attribute.map == {'chris': 'rich', 'fatema': 'poor', 'gerry': 'poor', 'fred': 'poor', 'nick': 'poor'}


def test_loading_v12_subpops(test_xml_config_v12, test_paths_v12):
    attribute = inputs.Subpopulations(test_xml_config_v12)
    attribute.build(test_paths_v12.resources)
    assert attribute.map == {'chris': 'rich', 'fatema': 'poor', 'gerry': 'poor', 'fred': 'poor', 'nick': 'poor'}


# Attributes
def test_loading_xml_attributes(test_xml_config, test_paths):
    attribute = inputs.Attributes(test_xml_config)
    attribute.build(test_paths.resources)
    assert len(attribute.map) == 5
    assert attribute.map['chris'] == {"subpopulation":"rich", "age": "yes"}


def test_loading_gzip_attributes(test_gzip_config, test_zip_paths):
    attribute = inputs.Attributes(test_gzip_config)
    attribute.build(test_zip_paths.resources)
    assert len(attribute.map) == 5
    assert attribute.map['chris'] == {"subpopulation":"rich", "age": "yes"}


def test_loading_v12_attributes(test_xml_config_v12, test_paths_v12):
    attribute = inputs.Attributes(test_xml_config_v12)
    attribute.build(test_paths_v12.resources)
    assert len(attribute.map) == 5
    assert attribute.map['chris'] == {"subpopulation":"rich", "age": "yes"}


# Output Config
def test_load_xml_output_config(test_xml_config, test_paths):
    out_config = inputs.OutputConfig(test_xml_config)
    out_config.build(test_paths.resources)
    assert set(out_config.activities) == set(['home', 'work'])
    assert set(out_config.modes) == set(['pt', 'walk', 'bike', 'car', 'transit_walk'])
    assert set(out_config.sub_populations) == set(['rich', 'poor', 'default'])


# Road Pricing
def test_load_road_pricing(test_xml_config, test_paths):
    road_pricing = inputs.RoadPricing(test_xml_config)
    road_pricing.build(test_paths.resources)
    assert set(road_pricing.links) == {"2-3", "3-4", "4-3", "2-1"}
    road_pricing.links == {
        '2-3': [
            {'start_time': '00:00', 'end_time': '06:00', 'amount': '0.0'},
            {'start_time': '06:01', 'end_time': '10:00', 'amount': '10.0'},
            {'start_time': '10:01', 'end_time': '16:00', 'amount': '3.0'},
            {'start_time': '16:01', 'end_time': '19:00', 'amount': '10.0'},
            {'start_time': '19:01', 'end_time': '23:59', 'amount': '3.0'}
        ],
        '3-4': [
            {'start_time': '00:00', 'end_time': '06:00', 'amount': '0.0'},
            {'start_time': '06:01', 'end_time': '10:00', 'amount': '10.0'},
            {'start_time': '10:01', 'end_time': '16:00', 'amount': '3.0'},
            {'start_time': '16:01', 'end_time': '19:00', 'amount': '10.0'},
            {'start_time': '19:01', 'end_time': '23:59', 'amount': '3.0'}
        ],
        '4-3': [
            {'start_time': '00:00', 'end_time': '06:00', 'amount': '0.0'},
            {'start_time': '06:01', 'end_time': '10:00', 'amount': '10.0'},
            {'start_time': '10:01', 'end_time': '16:00', 'amount': '3.0'},
            {'start_time': '16:01', 'end_time': '19:00', 'amount': '10.0'},
            {'start_time': '19:01', 'end_time': '23:59', 'amount': '3.0'}
        ],
        '2-1': [
            {'start_time': '00:00', 'end_time': '06:00', 'amount': '0.0'},
            {'start_time': '06:01', 'end_time': '10:00', 'amount': '10.0'},
            {'start_time': '10:01', 'end_time': '16:00', 'amount': '3.0'},
            {'start_time': '16:01', 'end_time': '19:00', 'amount': '10.0'},
            {'start_time': '19:01', 'end_time': '23:59', 'amount': '3.0'}
        ]
    }


# Input Manager
def test_load_input_manager(test_xml_config, test_paths):
    input_workstation = inputs.InputsWorkStation(test_xml_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    assert input_workstation.resources['events']
    assert input_workstation.resources['network']
    assert input_workstation.resources['transit_schedule']
    assert input_workstation.resources['transit_vehicles']
    assert input_workstation.resources['subpopulations']
    assert input_workstation.resources['attributes']
    assert input_workstation.resources['plans']
    assert input_workstation.resources['mode_map']
