import sys
import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree

# paths in config files etc. assume we're in the repo's root, so make sure we always are
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root_dir)

from elara.config import Config, PathFinderWorkStation
from elara import inputs
from elara import event_handlers
from elara.event_handlers import EventHandlerWorkStation, LinkVehicleCounts

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
test_inputs = os.path.join(test_dir, "test_intermediate_data")
test_outputs = os.path.join(test_dir, "test_outputs")
if not os.path.exists(test_outputs):
    os.mkdir(test_outputs)
benchmarks_path = os.path.join(test_outputs, "benchmarks")
if not os.path.exists(benchmarks_path):
    os.mkdir(benchmarks_path)


test_floor_data = [
    (0, 0),
    (1*3600, 1),
    (24*3600 - 1, 23),
    (24*3600, 0)
]


@pytest.mark.parametrize("seconds,hour", test_floor_data)
def test_table_position_floor(seconds, hour):
    assert event_handlers.table_position(
        elem_indices={1: 0},
        class_indices={1: 0},
        periods=24,
        elem_id=1,
        attribute_class=1,
        time=seconds
    )[-1] == hour


@pytest.fixture
def test_list():
    return [1, 2, 4]


@pytest.fixture
def test_df(test_list):
    array = np.ones((3, 24))
    array[2][0] = 0
    array[1] = 0
    index = test_list
    columns = [i for i in range(24)]
    df = pd.DataFrame(array, index=index, columns=columns)
    return df


# Config
@pytest.fixture
def test_config():
    config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
    config = Config(config_path)
    assert config
    return config


def test_tool_naming(test_config):
    tool = LinkVehicleCounts(config=test_config)
    assert (str(tool)) == "LinkVehicleCounts"


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
    input_workstation = inputs.InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()
    return input_workstation


# Base
@pytest.fixture
def base_handler(test_config, input_manager):
    base_handler = event_handlers.EventHandlerTool(test_config, 'car')
    assert base_handler.mode == 'car'
    base_handler.build(input_manager.resources)
    return base_handler


def test_build_list_indices(base_handler):
    test_list = [1, 2, 4]
    elems_in, elem_indices = base_handler.generate_elem_ids(test_list)
    assert elems_in == test_list
    assert elem_indices[test_list[-1]] == len(test_list) - 1


def test_build_df_indices(base_handler, test_df, test_list):
    elems_in, elem_indices = base_handler.generate_elem_ids(test_df)
    assert elems_in == test_list
    assert elem_indices[test_list[-1]] == len(test_list) - 1


def test_get_veh_mode(base_handler):
    assert base_handler.vehicle_mode('bus1').lower() == 'bus'
    assert base_handler.vehicle_mode('not_a_transit_vehicle') == "car"


def test_empty_rows(base_handler, test_df):
    assert len(base_handler.remove_empty_rows(test_df)) == 2


@pytest.fixture
def events(test_config, test_paths):
    events = inputs.Events(test_config)
    events.build(test_paths.resources)
    return events.elems


@pytest.fixture
def gerry_waiting_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23400.0" type="waitingForPt" agent="gerry" atStop="home_stop_out" 
        destinationStop="work_stop_in"  />
        """
    return etree.fromstring(string)


@pytest.fixture
def driver_enters_veh_event():
    time = 6.5 * 60 * 60 + (5 * 60)
    string = """
        <event time="23700.0" type="PersonEntersVehicle" person="pt_bus1_bus" vehicle="bus1"  />
        """
    return etree.fromstring(string)


@pytest.fixture
def gerry_enters_veh_event():
    time = 6.5 * 60 * 60 + (10 * 60)
    string = """
        <event time="24000.0" type="PersonEntersVehicle" person="gerry" vehicle="bus1"  />
        """
    return etree.fromstring(string)


@pytest.fixture
def veh_departs_event():
    time = 6.5 * 60 * 60 + (15 * 60)
    string = """
        <event time="24300.0" type="VehicleDepartsAtFacility" vehicle="bus1" 
        facility="home_stop_out" delay="0.0"  />
        """
    return etree.fromstring(string)


# Waiting times
@pytest.fixture
def test_agent_waiting_times_log_handler(test_config, input_manager):
    handler = event_handlers.StopPassengerWaiting(test_config, 'all')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    return handler


def test_agent_waiting_times_log_single_event_first_waiting(
        test_agent_waiting_times_log_handler,
        gerry_waiting_event,
        driver_enters_veh_event,
        gerry_enters_veh_event,
        veh_departs_event
):
    handler = test_agent_waiting_times_log_handler

    handler.process_event(gerry_waiting_event)
    assert len(handler.agent_status) == 1
    assert len(handler.veh_waiting_occupancy) == 0
    assert len(handler.waiting_time_log.chunk) == 0

    handler.process_event(driver_enters_veh_event)
    assert len(handler.agent_status) == 1
    assert len(handler.veh_waiting_occupancy) == 1
    assert len(handler.waiting_time_log.chunk) == 0

    handler.process_event(gerry_enters_veh_event)
    assert len(handler.agent_status) == 1
    assert len(handler.veh_waiting_occupancy) == 1
    assert len(handler.waiting_time_log.chunk) == 0

    handler.process_event(veh_departs_event)
    assert len(handler.agent_status) == 1
    assert len(handler.veh_waiting_occupancy) == 1
    assert len(handler.waiting_time_log.chunk) == 1


@pytest.fixture
def car_enters_link_event():
    time = 6.5 * 60 * 60
    string = """
        <event time = "23400.0" type = "vehicle enters traffic" person = "chris"
        link = "1-2" vehicle = "chris" networkMode = "car" relativePosition = "1.0" />
        """
    return etree.fromstring(string)

@pytest.fixture
def car_link_pair_event():
    time = 6.5 * 60 * 60
    string = """
        <events>
            <event time = "23400.0" type = "vehicle enters traffic" person = "chris"
            link = "1-2" vehicle = "chris" networkMode = "car" relativePosition = "1.0" />
            <event time = "23500.0" type = "left link" vehicle = "chris" link = "1-2" />
            <event time="23400.0" type="entered link" vehicle="nick" link="1-2" />
            <event time = "23450.0" type = "left link" vehicle = "nick" link="1-2" />
            <event time="23400.0" type="entered link" vehicle="sarah" link="1-2" />
            <event time = "23420.0" type = "left link" vehicle = "sarah" link="1-2" />
            <event time="23400.0" type="entered link" vehicle="fred" link="1-2" />
            <event time = "23425.0" type = "left link" vehicle = "fred" link="1-2" />
        </events>
        """
    return etree.fromstring(string)

@pytest.fixture
def bus_enters_link_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23400.0" type="entered link" vehicle="bus1" link="1-2" />
        """
    return etree.fromstring(string)

@pytest.fixture
def bus_leaves_link_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23450.0" type="left link" vehicle="bus1" link="1-2" />
        """
    return etree.fromstring(string)

# Volume Counts
# Car
@pytest.fixture
def test_car_volume_count_handler(test_config, input_manager):
    handler = event_handlers.LinkVehicleCounts(test_config, 'car')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert 'not_applicable' in handler.classes
    assert list(handler.class_indices.keys()) == handler.classes
    assert len(handler.elem_ids) == len(resources['network'].link_gdf)
    assert list(handler.elem_indices.keys()) == handler.elem_ids
    assert handler.counts.shape == (
        len(resources['network'].link_gdf), 3, periods)
    return handler


def test_volume_count_process_single_event_car(test_car_volume_count_handler, car_enters_link_event):
    handler = test_car_volume_count_handler
    elem = car_enters_link_event
    handler.process_event(elem)
    assert np.sum(handler.counts) == 1
    link_index = handler.elem_indices['1-2']
    class_index = handler.class_indices['rich']
    period = 6
    assert handler.counts[link_index][class_index][period] == 1


def test_volume_count_process_single_event_not_car(test_car_volume_count_handler, bus_enters_link_event):
    handler = test_car_volume_count_handler
    elem = bus_enters_link_event
    handler.process_event(elem)
    assert np.sum(handler.counts) == 0


def test_volume_count_process_events_car(test_car_volume_count_handler, events):
    handler = test_car_volume_count_handler
    for elem in events:
        handler.process_event(elem)
    assert np.sum(handler.counts) == 14
    link_index = handler.elem_indices['1-2']
    class_index = handler.class_indices['rich']
    period = 7
    assert handler.counts[link_index][class_index][period] == 1


def test_volume_count_finalise_car(test_car_volume_count_handler, events):
    handler = test_car_volume_count_handler
    for elem in events:
        handler.process_event(elem)
    handler.finalise()
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
            assert 'total' in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 14 / handler.config.scale_factor
        assert np.sum(df.values) == gdf.total.sum()
        if 'class' in gdf.columns:
            assert set(gdf.loc[:, 'class']) == set(handler.resources['subpopulations'].classes)


# bus
@pytest.fixture
def test_bus_volume_count_handler(test_config, input_manager):
    handler = event_handlers.LinkVehicleCounts(test_config, 'bus')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert 'not_applicable' in handler.classes
    3
    assert list(handler.class_indices.keys()) == handler.classes
    assert len(handler.elem_ids) == len(resources['network'].link_gdf)
    assert list(handler.elem_indices.keys()) == handler.elem_ids
    assert handler.counts.shape == (
        len(resources['network'].link_gdf), 3, periods)
    return handler


def test_volume_count_process_single_event_bus(
        test_bus_volume_count_handler,
        bus_enters_link_event
):
    handler = test_bus_volume_count_handler
    elem = bus_enters_link_event
    handler.process_event(elem)
    assert np.sum(handler.counts) == 1
    link_index = handler.elem_indices['1-2']
    class_index = handler.class_indices['not_applicable']
    period = 6
    assert handler.counts[link_index][class_index][period] == 1


def test_volume_count_process_single_event_not_bus(
        test_bus_volume_count_handler,
        car_enters_link_event
):
    handler = test_bus_volume_count_handler
    elem = car_enters_link_event
    handler.process_event(elem)
    assert np.sum(handler.counts) == 0


def test_volume_count_process_events_bus(test_bus_volume_count_handler, events):
    handler = test_bus_volume_count_handler
    for elem in events:
        handler.process_event(elem)
    assert np.sum(handler.counts) == 12
    link_index = handler.elem_indices['1-2']
    class_index = handler.class_indices['not_applicable']
    period = 7
    assert handler.counts[link_index][class_index][period] == 1


def test_volume_count_finalise_bus(test_bus_volume_count_handler, events):
    handler = test_bus_volume_count_handler
    for elem in events:
        handler.process_event(elem)

    assert handler.mode.lower() == 'bus'
    assert handler.config.scale_factor == 0.0001

    handler.finalise()
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
            assert 'total' in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 12
        assert np.sum(df.values) == gdf.total.sum()
        if 'class' in gdf.columns:
            assert set(gdf.loc[:, 'class']) == set(handler.resources['subpopulations'].classes)

# Link Speeds
# Car
@pytest.fixture
def test_car_link_speed_handler(test_config, input_manager):
    handler = event_handlers.LinkVehicleSpeeds(test_config, 'car')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert 'not_applicable' in handler.classes
    assert list(handler.class_indices.keys()) == handler.classes
    assert len(handler.elem_ids) == len(resources['network'].link_gdf)
    assert list(handler.elem_indices.keys()) == handler.elem_ids
    assert handler.counts.shape == (
        len(resources['network'].link_gdf), 3, periods)
    return handler


def test_link_speed_process_single_event_car(test_car_link_speed_handler, car_enters_link_event):
    handler = test_car_link_speed_handler
    elem = car_enters_link_event
    handler.process_event(elem)
    assert np.sum(handler.counts) == 0 #shouldnt add count for entering event, only leaving
    link_index = handler.elem_indices['1-2']
    class_index = handler.class_indices['rich']
    period = 6
    assert handler.counts[link_index][class_index][period] == 0

def test_link_speed_process_single_event_not_car(test_car_link_speed_handler, bus_enters_link_event):
    handler = test_car_link_speed_handler
    elem = bus_enters_link_event
    handler.process_event(elem)
    assert np.sum(handler.counts) == 0


def test_link_speed_process_events_car(test_car_link_speed_handler, car_link_pair_event):
    handler = test_car_link_speed_handler
    for elem in car_link_pair_event:
        handler.process_event(elem)
    assert np.sum(handler.counts) == 3
    assert np.sum(handler.inverseduration_sum)== 0.11
    link_index = handler.elem_indices['1-2']
    class_index = handler.class_indices['poor']
    period = 6
    assert handler.counts[link_index][class_index][period] == 2
    assert handler.inverseduration_sum[link_index][class_index][period] == 1/50+1/25


def test_link_speed_finalise_car(test_car_link_speed_handler, car_link_pair_event):
    handler = test_car_link_speed_handler
    for elem in car_link_pair_event:
        handler.process_event(elem)
    handler.finalise()
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
        df = gdf.loc[:, cols]
        if name == "link_vehicle_speeds_car_average":
            assert np.sum(df.values) == (2+5+4)/3
        elif name == "link_vehicle_speeds_car_min":
            assert np.sum(df.values) == 2
        elif name == "link_vehicle_speeds_car_max":
            assert np.sum(df.values) == 5


# Passenger Counts Handler Tests
@pytest.fixture
def bus_passenger_count_handler(test_config, input_manager):
    handler = event_handlers.LinkPassengerCounts(test_config, 'bus')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert 'not_applicable' in handler.classes
    assert list(handler.class_indices.keys()) == handler.classes
    assert handler.counts.shape == (
        len(resources['network'].link_gdf), 3, periods)
    return handler


# Route Passenger Counts Handler Tests
@pytest.fixture
def bus_route_passenger_count_handler(test_config, input_manager):
    handler = event_handlers.RoutePassengerCounts(test_config, 'bus')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert 'not_applicable' in handler.classes
    assert list(handler.class_indices.keys()) == handler.classes
    assert handler.counts.shape == (
        len(set(resources['transit_schedule'].veh_to_route_map.values())), 3, periods)
    return handler


@pytest.fixture
def driver_enters_veh_event():
    string = """
        <event time="23400.0" type="PersonEntersVehicle" person="pt_bus1_bus" vehicle="bus1" />
        """
    return etree.fromstring(string)


@pytest.fixture
def person_enters_veh_event():
    string = """
        <event time="23401.0" type="PersonEntersVehicle" person="gerry" vehicle="bus1"  />
        """
    return etree.fromstring(string)


@pytest.fixture
def person2_enters_veh_event():
    string = """
        <event time="23401.0" type="PersonEntersVehicle" person="chris" vehicle="bus1"  />
        """
    return etree.fromstring(string)


@pytest.fixture
def bus_leaves_link_event():
    string = """
        <event time="23405.0" type="left link" vehicle="bus1" link="2-3"  />
        """
    return etree.fromstring(string)


@pytest.fixture
def person_leaves_veh_event():
    string = """
        <event time="23410.0" type="PersonLeavesVehicle" person="gerry" vehicle="bus1" />
        """
    return etree.fromstring(string)


@pytest.fixture
def person2_leaves_veh_event():
    string = """
        <event time="23410.0" type="PersonLeavesVehicle" person="chris" vehicle="bus1" />
        """
    return etree.fromstring(string)


def test_passenger_count_process_single_event_driver(
        bus_passenger_count_handler,
        driver_enters_veh_event):
    handler = bus_passenger_count_handler
    elem = driver_enters_veh_event
    handler.process_event(elem)
    assert sum(handler.veh_occupancy.values()) == 0
    assert np.sum(handler.counts) == 0


def test_passenger_count_process_single_event_link(
        bus_passenger_count_handler,
        bus_leaves_link_event):
    handler = bus_passenger_count_handler
    elem = bus_leaves_link_event
    handler.process_event(elem)
    assert sum(handler.veh_occupancy.values()) == 0
    assert np.sum(handler.counts) == 0


def test_passenger_count_process_single_event_passenger(
        bus_passenger_count_handler,
        person_enters_veh_event):
    handler = bus_passenger_count_handler
    elem = person_enters_veh_event
    handler.process_event(elem)
    assert sum([v for vo in handler.veh_occupancy.values() for v in vo.values()]) == 1
    assert np.sum(handler.counts) == 0


def test_passenger_count_process_events(
        bus_passenger_count_handler,
        person_enters_veh_event,
        person2_enters_veh_event,
        car_enters_link_event,
        bus_leaves_link_event):
    handler = bus_passenger_count_handler
    handler.process_event(person_enters_veh_event)
    handler.process_event(person2_enters_veh_event)
    handler.process_event(car_enters_link_event)
    assert sum([v for vo in handler.veh_occupancy.values() for v in vo.values()]) == 2
    handler.process_event(bus_leaves_link_event)
    link_index = handler.elem_indices['2-3']
    class_index = handler.class_indices['rich']
    period = 6
    assert np.sum(handler.counts[link_index, :, :]) == 2
    assert np.sum(handler.counts[:, :, period]) == 2
    assert handler.counts[link_index, class_index, period] == 1


def test_passenger_count_finalise_bus(
        bus_passenger_count_handler,
        events
):
    handler = bus_passenger_count_handler
    for elem in events:
        handler.process_event(elem)
    handler.finalise()
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
            assert 'total' in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 8 / handler.config.scale_factor
        assert np.sum(df.values) == gdf.total.sum()
        if 'class' in gdf.columns:
            assert set(gdf.loc[:, 'class']) == set(handler.resources['subpopulations'].classes)


def test_route_passenger_count_handler_rejects_car_as_mode():
    with pytest.raises(UserWarning) as ex_info:
        event_handlers.RoutePassengerCounts(config=test_config, mode='car')
    assert "Invalid mode option: car at tool" in str(ex_info.value)


def test_route_passenger_count_finalise_bus(bus_route_passenger_count_handler, events):
    for elem in events:
        bus_route_passenger_count_handler.process_event(elem)
    bus_route_passenger_count_handler.finalise()

    for name, gdf in bus_route_passenger_count_handler.result_dfs.items():
        cols = list(range(bus_route_passenger_count_handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 8 / bus_route_passenger_count_handler.config.scale_factor
        if 'class' in gdf.columns:
            assert set(gdf.loc[:, 'class']) == set(bus_route_passenger_count_handler.resources['subpopulations'].classes)


# Stop Interactions
@pytest.fixture
def test_bus_passenger_interaction_handler(test_config, input_manager):
    handler = event_handlers.StopPassengerCounts(test_config, 'bus')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert 'not_applicable' in handler.classes
    3
    assert list(handler.class_indices.keys()) == handler.classes
    assert handler.boardings.shape == (
        len(resources['transit_schedule'].stop_gdf),
        3,
        periods
    )
    return handler


@pytest.fixture
def waiting_for_pt_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23400.0" type="waitingForPt" agent="gerry" 
        atStop="home_stop_out" destinationStop="work_stop_in" />
        """
    return etree.fromstring(string)


@pytest.fixture
def waiting2_for_pt_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23401.0" type="waitingForPt" agent="chris" 
        atStop="home_stop_out" destinationStop="work_stop_in" />
        """
    return etree.fromstring(string)


def test_stop_interaction_process_single_event_driver(
        test_bus_passenger_interaction_handler,
        driver_enters_veh_event):
    handler = test_bus_passenger_interaction_handler
    elem = driver_enters_veh_event
    handler.process_event(elem)
    assert sum(handler.agent_status.values()) == 0
    assert np.sum(handler.boardings) == 0
    assert np.sum(handler.alightings) == 0


def test_stop_interaction_process_single_event_link(
        test_bus_passenger_interaction_handler,
        bus_leaves_link_event):
    handler = test_bus_passenger_interaction_handler
    elem = bus_leaves_link_event
    handler.process_event(elem)
    assert sum(handler.agent_status.values()) == 0
    assert np.sum(handler.boardings) == 0
    assert np.sum(handler.alightings) == 0


def test_stop_interaction_process_single_event_passenger(
        test_bus_passenger_interaction_handler,
        waiting_for_pt_event):
    handler = test_bus_passenger_interaction_handler
    elem = waiting_for_pt_event
    handler.process_event(elem)
    assert len(handler.agent_status.values()) == 1
    assert np.sum(handler.boardings) == 0
    assert np.sum(handler.alightings) == 0


def test_stop_interaction_process_events(
        test_bus_passenger_interaction_handler,
        waiting_for_pt_event,
        waiting2_for_pt_event,
        person_enters_veh_event,
        person2_enters_veh_event,
        person_leaves_veh_event,
):
    handler = test_bus_passenger_interaction_handler
    handler.process_event(waiting_for_pt_event)
    handler.process_event(waiting2_for_pt_event)
    assert len(handler.agent_status.values()) == 2
    handler.process_event(person_enters_veh_event)
    handler.process_event(person2_enters_veh_event)
    handler.process_event(person_leaves_veh_event)
    assert len(handler.agent_status.values()) == 1
    link_index = handler.elem_indices['home_stop_out']
    class_index = handler.class_indices['rich']
    period = 6
    assert np.sum(handler.boardings[link_index, :, :]) == 2
    assert np.sum(handler.boardings[:, :, period]) == 2
    assert handler.boardings[link_index, class_index, period] == 1
    link_index = handler.elem_indices['work_stop_in']
    class_index = handler.class_indices['poor']
    period = 6
    assert np.sum(handler.alightings[link_index, :, :]) == 1
    assert np.sum(handler.alightings[:, :, period]) == 1
    assert handler.alightings[link_index, class_index, period] == 1


def test_stop_interaction_finalise_bus(
        test_bus_passenger_interaction_handler,
        events
):
    handler = test_bus_passenger_interaction_handler
    for elem in events:
        handler.process_event(elem)
    handler.finalise()
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
            assert 'total' in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 4 / handler.config.scale_factor
        assert np.sum(df.values) == gdf.total.sum()
        if 'class' in gdf.columns:
            assert set(gdf.loc[:, 'class']) == set(handler.resources['subpopulations'].classes)


# Stop to stop volumes
@pytest.fixture
def bus_stop_to_stop_handler(test_config, input_manager):
    handler = event_handlers.StopToStopPassengerCounts(test_config, 'bus')
    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    return handler


@pytest.fixture
def test_bus_stop_to_stop_handler(bus_stop_to_stop_handler, input_manager):
    handler = bus_stop_to_stop_handler
    resources = input_manager.resources
    periods = 24
    assert 'not_applicable' in handler.classes
    3
    assert list(handler.class_indices.keys()) == handler.classes
    assert handler.counts.shape == (
        len(resources['transit_schedule'].stop_gdf),
        len(resources['transit_schedule'].stop_gdf),
        3,
        periods
    )
    return handler


@pytest.fixture
def veh_arrives_facilty_event1():
    time = 6.5 * 60 * 60
    string = f"""
        <event time="{time}" type="VehicleArrivesAtFacility" vehicle="bus1" facility="home_stop_out" delay="Infinity"/>
        """
    return etree.fromstring(string)


@pytest.fixture
def veh_arrives_facilty_event2():
    time = 7.5 * 60 * 60
    string = f"""
        <event time="{time}" type="VehicleArrivesAtFacility" vehicle="bus1" facility="work_stop_in" delay="Infinity"/>
        """
    return etree.fromstring(string)


@pytest.fixture
def veh_arrives_facilty_event3():
    time = 8.5 * 60 * 60
    string = f"""
        <event time="{time}" type="VehicleArrivesAtFacility" vehicle="bus1" facility="home_stop_out" delay="Infinity"/>
        """
    return etree.fromstring(string)


def test_veh_tracker_events(
    bus_stop_to_stop_handler,
    veh_arrives_facilty_event1,
    veh_arrives_facilty_event2
    ):
    handler = bus_stop_to_stop_handler
    handler.process_event(veh_arrives_facilty_event1)
    assert handler.veh_tracker == {"bus1":"home_stop_out"}
    handler.process_event(veh_arrives_facilty_event2)
    assert handler.veh_tracker == {"bus1":"work_stop_in"}

def test_stop_to_stop_veh_interaction_process_single_event_driver(
        bus_stop_to_stop_handler,
        veh_arrives_facilty_event1,
        driver_enters_veh_event,
        ):
    handler = bus_stop_to_stop_handler
    handler.process_event(veh_arrives_facilty_event1)
    handler.process_event(driver_enters_veh_event)
    assert handler.veh_occupancy == {}
    assert np.sum(handler.counts) == 0


def test_stop_to_stop_process_events(
        bus_stop_to_stop_handler,
        veh_arrives_facilty_event1,
        person_enters_veh_event,
        person2_enters_veh_event,
        veh_arrives_facilty_event2,
        person_leaves_veh_event,
        veh_arrives_facilty_event3
        ):
    handler = bus_stop_to_stop_handler
    handler.process_event(veh_arrives_facilty_event1)
    handler.process_event(person_enters_veh_event)
    handler.process_event(person2_enters_veh_event)
    handler.process_event(veh_arrives_facilty_event2)

    assert handler.veh_occupancy == {'bus1': {'poor': 1, 'rich': 1}}
    assert np.sum(handler.counts) == 2

    handler.process_event(person_leaves_veh_event)
    handler.process_event(veh_arrives_facilty_event3)

    assert handler.veh_occupancy == {'bus1': {'poor': 0, 'rich': 1}}
    assert np.sum(handler.counts) == 3

    stop_index1 = handler.elem_indices['home_stop_out']
    stop_index2 = handler.elem_indices['work_stop_in']
    stop_index3 = handler.elem_indices['home_stop_in']
    class_index = handler.class_indices['rich']
    period = 7

    assert np.sum(handler.counts[stop_index1, stop_index2, :, :]) == 2
    assert np.sum(handler.counts[stop_index1, stop_index2, class_index, :]) == 1
    assert np.sum(handler.counts[:, :, :, period]) == 2

    period = 8

    assert np.sum(handler.counts[stop_index2, stop_index1, :, :]) == 1
    assert np.sum(handler.counts[:, :, class_index, :]) == 2
    assert np.sum(handler.counts[:, :, :, period]) == 1


def test_stop_to_stop_finalise_bus(
        bus_stop_to_stop_handler,
        veh_arrives_facilty_event1,
        person_enters_veh_event,
        person2_enters_veh_event,
        veh_arrives_facilty_event2,
        person_leaves_veh_event,
        veh_arrives_facilty_event3
):
    handler = bus_stop_to_stop_handler
    events = [
        veh_arrives_facilty_event1,
        person_enters_veh_event,
        person2_enters_veh_event,
        veh_arrives_facilty_event2,
        person_leaves_veh_event,
        veh_arrives_facilty_event3
    ]
    for elem in events:
        handler.process_event(elem)
    
    handler.finalise()
    gdf = handler.result_dfs["stop_to_stop_passenger_counts_bus"]
    cols = list(range(handler.config.time_periods))
    for c in cols:
        assert c in gdf.columns
    df = gdf.loc[:, cols]
    assert np.sum(df.values) == 3 / handler.config.scale_factor
    if 'class' in gdf.columns:
        assert set(gdf.loc[:, 'class']) == set(handler.resources['subpopulations'].classes)

    gdf = handler.result_dfs["stop_to_stop_passenger_counts_bus_total"]
    cols = list(range(handler.config.time_periods))
    for c in cols:
        assert c in gdf.columns
        assert 'total' in gdf.columns
    df = gdf.loc[:, cols]
    assert np.sum(df.values) == 3 / handler.config.scale_factor
    assert np.sum(df.values) == gdf.total.sum()
    if 'class' in gdf.columns:
        assert set(gdf.loc[:, 'class']) == set(handler.resources['subpopulations'].classes)


# Event Handler Manager
def test_load_event_handler_manager(test_config, test_paths):
    input_workstation = inputs.InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    event_workstation = EventHandlerWorkStation(test_config)
    event_workstation.connect(managers=None, suppliers=[input_workstation])
    event_workstation.load_all_tools(mode='bus')
    event_workstation.build(write_path=test_outputs)

    for handler_name, handler in event_workstation.resources.items():
        for name, gdf in handler.result_dfs.items():
            cols = list(range(handler.config.time_periods))
            for c in cols:
                assert c in gdf.columns
            df = gdf.loc[:, cols]
            assert np.sum(df.values)

