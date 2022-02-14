import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree

from elara.config import Config, PathFinderWorkStation
from elara import inputs
from elara import event_handlers
from elara.event_handlers import EventHandlerWorkStation, LinkVehicleCounts

# paths in config files etc. assume we're in the repo's root, so make sure we always are
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root_dir)

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
    assert (str(tool)) == "LinkVehicleCountsAll"


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
# subpopulation breakdown
@pytest.fixture
def test_car_volume_count_handler_with_subpopulations(test_config, input_manager):
    handler = event_handlers.LinkVehicleCounts(test_config, mode='car', groupby_person_attribute="subpopulation")

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert None in handler.classes
    assert set(handler.class_indices.keys()) == handler.classes
    assert len(handler.elem_ids) == len(resources['network'].link_gdf)
    assert list(handler.elem_indices.keys()) == handler.elem_ids
    assert handler.counts.shape == (
        len(resources['network'].link_gdf), 3, periods)
    return handler


def test_volume_count_process_single_event_car(test_car_volume_count_handler_with_subpopulations, car_enters_link_event):
    handler = test_car_volume_count_handler_with_subpopulations
    elem = car_enters_link_event
    handler.process_event(elem)
    assert np.sum(handler.counts) == 1
    link_index = handler.elem_indices['1-2']
    class_index = handler.class_indices['rich']
    period = 6
    assert handler.counts[link_index][class_index][period] == 1


def test_volume_count_process_single_event_not_car(test_car_volume_count_handler_with_subpopulations, bus_enters_link_event):
    handler = test_car_volume_count_handler_with_subpopulations
    elem = bus_enters_link_event
    handler.process_event(elem)
    assert np.sum(handler.counts) == 0


def test_volume_count_process_events_car(test_car_volume_count_handler_with_subpopulations, events):
    handler = test_car_volume_count_handler_with_subpopulations
    for elem in events:
        handler.process_event(elem)
    assert np.sum(handler.counts) == 14
    link_index = handler.elem_indices['1-2']
    class_index = handler.class_indices['rich']
    period = 7
    assert handler.counts[link_index][class_index][period] == 1


def test_volume_count_finalise_car(test_car_volume_count_handler_with_subpopulations, events):
    handler = test_car_volume_count_handler_with_subpopulations
    for elem in events:
        handler.process_event(elem)
    handler.finalise()
    assert len(handler.result_dfs) == 2
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
            assert 'total' in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 14 / handler.config.scale_factor
        assert np.sum(df.values) == gdf.total.sum()
        if 'subpopulation' in gdf.columns:
            assert set(gdf.loc[:, 'subpopulation']) == {"poor", "rich", np.nan}


# No class breakdown
@pytest.fixture
def test_car_volume_count_handler_simple(test_config, input_manager):
    handler = event_handlers.LinkVehicleCounts(test_config, mode='car')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert None in handler.classes
    assert set(handler.class_indices.keys()) == handler.classes
    assert len(handler.elem_ids) == len(resources['network'].link_gdf)
    assert list(handler.elem_indices.keys()) == handler.elem_ids
    assert handler.counts.shape == (
        len(resources['network'].link_gdf), 1, periods)
    return handler


def test_volume_count_finalise_car_simple(test_car_volume_count_handler_simple, events):
    handler = test_car_volume_count_handler_simple
    for elem in events:
        handler.process_event(elem)
    handler.finalise()
    assert len(handler.result_dfs) == 1
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
            assert 'total' in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 14 / handler.config.scale_factor
        assert np.sum(df.values) == gdf.total.sum()


# bus
@pytest.fixture
def test_bus_volume_count_handler(test_config, input_manager):
    handler = event_handlers.LinkVehicleCounts(test_config, mode='bus', groupby_person_attribute="subpopulation")

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert None in handler.classes
    assert set(handler.class_indices.keys()) == handler.classes
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
    class_index = handler.class_indices[None]
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
    class_index = handler.class_indices[None]
    period = 7
    assert handler.counts[link_index][class_index][period] == 1


def test_volume_count_finalise_bus(test_bus_volume_count_handler, events):
    handler = test_bus_volume_count_handler
    for elem in events:
        handler.process_event(elem)

    assert handler.mode.lower() == 'bus'
    assert handler.config.scale_factor == 0.0001

    handler.finalise()
    assert len(handler.result_dfs) == 2
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
            assert 'total' in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 12
        assert np.sum(df.values) == gdf.total.sum()
        if 'subpopulation' in gdf.columns:
            assert set(gdf.loc[:, 'subpopulation']) == {"poor", "rich", np.nan}

# Link Speeds
# Car
# With subpopulation attribute groups
@pytest.fixture
def test_car_link_speed_handler(test_config, input_manager):
    handler = event_handlers.LinkVehicleSpeeds(test_config, mode='car', groupby_person_attribute="subpopulation")

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert None in handler.classes
    assert set(handler.class_indices.keys()) == {'poor', 'rich', None}
    assert len(handler.elem_ids) == len(resources['network'].link_gdf)
    assert list(handler.elem_indices.keys()) == handler.elem_ids
    assert handler.counts.shape == (
        len(resources['network'].link_gdf), 3, periods)
    return handler


def test_link_speed_process_single_event_car(test_car_link_speed_handler, car_enters_link_event):
    handler = test_car_link_speed_handler
    elem = car_enters_link_event
    handler.process_event(elem)
    assert np.sum(handler.counts) == 0 # shouldnt add count for entering event, only leaving
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
    assert len(handler.result_dfs) == 6
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
        df = gdf.loc[:, cols]
        if name == "link_vehicle_speeds_car_average":
            assert np.sum(df.values) == (2+5+4)/3 * 3.6
        elif name == "link_vehicle_speeds_car_average_subpopulation":
            assert np.sum(df.values) == (2+5+4) * 3.6
        elif name == "link_vehicle_speeds_car_min":
            assert np.sum(df.values) == 2 * 3.6
        elif name == "link_vehicle_speeds_car_min_subpopulation": # TODO something wrong here
            assert np.sum(df.values) == 7 * 3.6
        elif name == "link_vehicle_speeds_car_max":
            assert np.sum(df.values) == 5 * 3.6
        elif name == "link_vehicle_speeds_car_max_subpopulation":
            assert np.sum(df.values) == 9 * 3.6


# With no attribute groups
@pytest.fixture
def test_car_link_speed_handler_simple(test_config, input_manager):
    handler = event_handlers.LinkVehicleSpeeds(test_config, mode='car', groupby_person_attribute=None)

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert None in handler.classes
    assert set(handler.class_indices.keys()) == {None}
    assert len(handler.elem_ids) == len(resources['network'].link_gdf)
    assert list(handler.elem_indices.keys()) == handler.elem_ids
    assert handler.counts.shape == (
        len(resources['network'].link_gdf), 1, periods)
    return handler


def test_link_speed_finalise_car_simple(test_car_link_speed_handler_simple, car_link_pair_event):
    handler = test_car_link_speed_handler_simple
    for elem in car_link_pair_event:
        handler.process_event(elem)
    handler.finalise()
    assert len(handler.result_dfs) == 3
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
        # df = gdf.loc[:, cols]
        # if name == "link_vehicle_speeds_car_average":
        #     assert np.sum(df.values) == (2+5+4)/3
        # elif name == "link_vehicle_speeds_car_min":
        #     assert np.sum(df.values) == 2
        # elif name == "link_vehicle_speeds_car_max":
        #     assert np.sum(df.values) == 5


# Passenger Counts Handler Tests
# with subpopulation attribute grouping
@pytest.fixture
def bus_passenger_count_handler(test_config, input_manager):
    handler = event_handlers.LinkPassengerCounts(test_config, mode='bus', groupby_person_attribute="subpopulation")

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    periods = 24
    assert None in handler.classes
    assert set(handler.class_indices.keys()) == {'poor', 'rich', None}
    assert handler.counts.shape == (
        len(resources['network'].link_gdf), 3, periods)
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

@pytest.fixture
def person_enters_veh2_event():
    string = """
        <event time="23501.0" type="PersonEntersVehicle" person="gerry" vehicle="bus2"  />
        """
    return etree.fromstring(string)

def person_leaves_veh2_event():
    string = """
        <event time="24501.0" type="PersonLeavesVehicle" person="gerry" vehicle="bus2"  />
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
    assert len(handler.result_dfs) == 2
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
            assert 'total' in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 8 / handler.config.scale_factor
        assert np.sum(df.values) == gdf.total.sum()
        if 'subpopulation' in gdf.columns:
            assert set(gdf.loc[:, 'subpopulation']) == {"poor", "rich", np.nan}


def test_route_passenger_count_handler_rejects_car_as_mode():
    with pytest.raises(UserWarning) as ex_info:
        event_handlers.RoutePassengerCounts(config=test_config, mode='car')
    assert "Invalid mode option: car at tool" in str(ex_info.value)


# Route Passenger Counts Handler Tests
# with subpopulation attribute grouping
@pytest.fixture
def bus_route_passenger_count_handler(test_config, input_manager):
    handler = event_handlers.RoutePassengerCounts(test_config, mode='bus', groupby_person_attribute="subpopulation")

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert None in handler.classes
    assert set(handler.class_indices.keys()) == {'poor', 'rich', None}
    assert handler.counts.shape == (
        len(set(resources['transit_schedule'].veh_to_route_map.values())), 3, periods)
    return handler


def test_route_passenger_count_finalise_bus(bus_route_passenger_count_handler, events):
    for elem in events:
        bus_route_passenger_count_handler.process_event(elem)
    bus_route_passenger_count_handler.finalise()

    assert len(bus_route_passenger_count_handler.result_dfs) == 2

    for name, gdf in bus_route_passenger_count_handler.result_dfs.items():
        cols = list(range(bus_route_passenger_count_handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 8 / bus_route_passenger_count_handler.config.scale_factor
        if 'subpopulation' in gdf.columns:
            assert set(gdf.loc[:, 'subpopulation']) == {'poor', 'rich', np.nan}


# simple case no attribute breakdown
@pytest.fixture
def bus_route_passenger_count_handler_simple(test_config, input_manager):
    handler = event_handlers.RoutePassengerCounts(test_config, mode='bus', groupby_person_attribute=None)

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert None in handler.classes
    assert set(handler.class_indices.keys()) == {None}
    assert handler.counts.shape == (
        len(set(resources['transit_schedule'].veh_to_route_map.values())), 1, periods)
    return handler


def test_route_passenger_count_finalise_bus_simple(bus_route_passenger_count_handler_simple, events):
    for elem in events:
        bus_route_passenger_count_handler_simple.process_event(elem)
    bus_route_passenger_count_handler_simple.finalise()

    assert len(bus_route_passenger_count_handler_simple.result_dfs) == 1

    for name, gdf in bus_route_passenger_count_handler_simple.result_dfs.items():
        cols = list(range(bus_route_passenger_count_handler_simple.config.time_periods))
        for c in cols:
            assert c in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 8 / bus_route_passenger_count_handler_simple.config.scale_factor
        if 'subpopulation' in gdf.columns:
            assert set(gdf.loc[:, 'subpopulation']) == {'poor', 'rich', np.nan}


# Stop Interactions
# with subpopulation attribute grouping
@pytest.fixture
def test_bus_passenger_interaction_handler(test_config, input_manager):
    handler = event_handlers.StopPassengerCounts(test_config, mode='bus', groupby_person_attribute="subpopulation")

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert None in handler.classes
    assert set(handler.class_indices.keys()) == {'poor', 'rich', None}
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
        if 'subpopulation' in gdf.columns:
            assert set(gdf.loc[:, 'subpopulation']) == {'poor', 'rich', np.nan}

# simple case no attribute breakdown
@pytest.fixture
def test_bus_passenger_interaction_handler_simple(test_config, input_manager):
    handler = event_handlers.StopPassengerCounts(test_config, mode='bus', groupby_person_attribute=None)

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    assert None in handler.classes
    assert set(handler.class_indices.keys()) == {None}
    assert handler.boardings.shape == (
        len(resources['transit_schedule'].stop_gdf),
        1,
        periods
    )
    return handler


def test_stop_interaction_finalise_bus_simple(
        test_bus_passenger_interaction_handler_simple,
        events
):
    handler = test_bus_passenger_interaction_handler_simple
    for elem in events:
        handler.process_event(elem)
    handler.finalise()
    assert len(handler.result_dfs) == 2
    for name, gdf in handler.result_dfs.items():
        cols = list(range(handler.config.time_periods))
        for c in cols:
            assert c in gdf.columns
            assert 'total' in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 4 / handler.config.scale_factor
        assert np.sum(df.values) == gdf.total.sum()


# Stop to stop volumes
# with subpopulation attribute groups
@pytest.fixture
def bus_stop_to_stop_handler(test_config, input_manager):
    handler = event_handlers.StopToStopPassengerCounts(test_config, mode='bus', groupby_person_attribute="subpopulation")
    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    return handler

@pytest.fixture
def bus_vehicle_stop_to_stop_handler(test_config, input_manager):
    handler = event_handlers.VehicleStopToStopPassengerCounts(test_config, mode='bus', groupby_person_attribute="subpopulation")
    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    return handler


@pytest.fixture
def test_bus_stop_to_stop_handler(bus_stop_to_stop_handler, input_manager):
    handler = bus_stop_to_stop_handler
    resources = input_manager.resources
    periods = 24
    assert None in handler.classes
    assert list(handler.class_indices.keys()) == {"rich", "poor", None}
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

@pytest.fixture
def veh2_arrives_facilty_event1():
    time = 6.5 * 60 * 60
    string = f"""
        <event time="{time}" type="VehicleArrivesAtFacility" vehicle="bus2" facility="home_stop_out" delay="Infinity"/>
        """
    return etree.fromstring(string)


@pytest.fixture
def veh2_arrives_facilty_event2():
    time = 7.5 * 60 * 60
    string = f"""
        <event time="{time}" type="VehicleArrivesAtFacility" vehicle="bus2" facility="work_stop_in" delay="Infinity"/>
        """
    return etree.fromstring(string)


@pytest.fixture
def veh2_arrives_facilty_event3():
    time = 8.5 * 60 * 60
    string = f"""
        <event time="{time}" type="VehicleArrivesAtFacility" vehicle="bus2" facility="home_stop_out" delay="Infinity"/>
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

def test_vehicle_stop_to_stop_process_events(
        bus_vehicle_stop_to_stop_handler,
        veh_arrives_facilty_event1,
        veh2_arrives_facilty_event1,
        person_enters_veh_event,
        person2_enters_veh_event,
        veh_arrives_facilty_event2,
        veh2_arrives_facilty_event2,
        person_leaves_veh_event,
        veh_arrives_facilty_event3,
        veh2_arrives_facilty_event3,
        person_enters_veh2_event
        ):
    handler = bus_vehicle_stop_to_stop_handler
    handler.process_event(veh_arrives_facilty_event1)
    handler.process_event(veh2_arrives_facilty_event1)
    handler.process_event(person_enters_veh_event)
    handler.process_event(person2_enters_veh_event)
    handler.process_event(veh_arrives_facilty_event2)
    handler.process_event(veh2_arrives_facilty_event2)

    veh_index = handler.veh_ids_indices['bus1']
    veh2_index = handler.veh_ids_indices['bus2']
    assert handler.veh_occupancy == {'bus1': {'poor': 1, 'rich': 1}}
    assert sum(handler.counts.values()) == 2

    handler.process_event(person_leaves_veh_event)
    handler.process_event(person_enters_veh2_event) # person 1 interchanges from bus1 to bus2
    handler.process_event(veh_arrives_facilty_event3)
    handler.process_event(veh2_arrives_facilty_event3)
    counts_ser = pd.Series(handler.counts)

    assert handler.veh_occupancy == {'bus1': {'poor': 0, 'rich': 1}, 'bus2': {'poor': 1}}
    assert sum(handler.counts.values()) == 4

    period = 7
    assert np.sum(counts_ser.loc['home_stop_out', 'work_stop_in', 'bus1', :, :]) == 2
    assert np.sum(counts_ser.loc['home_stop_out', 'work_stop_in', 'bus1', 'rich', :]) == 1
    assert np.sum(counts_ser.loc[:, :, 'bus1', :, period]) == 2

    period = 8
    assert np.sum(counts_ser.loc['work_stop_in', 'home_stop_out', 'bus1', :, :]) == 1
    assert np.sum(counts_ser.loc['work_stop_in', 'home_stop_out', 'bus2', :, :]) == 1
    assert np.sum(counts_ser.loc[:, :, 'bus1', 'rich', :]) == 2
    assert np.sum(counts_ser.loc[:, :, 'bus1', :, period]) == 1
    assert np.sum(counts_ser.loc[:, :, :, :, period]) == 2

def test_vehicle_stop_to_stop_finalise_bus(
        bus_vehicle_stop_to_stop_handler,
        veh_arrives_facilty_event1,
        veh2_arrives_facilty_event1,
        person_enters_veh_event,
        person2_enters_veh_event,
        veh_arrives_facilty_event2,
        veh2_arrives_facilty_event2,
        person_leaves_veh_event,
        person_enters_veh2_event,
        veh_arrives_facilty_event3,
        veh2_arrives_facilty_event3
):
    handler = bus_vehicle_stop_to_stop_handler
    events = [
        veh_arrives_facilty_event1,
        veh2_arrives_facilty_event1,
        person_enters_veh_event,
        person2_enters_veh_event,
        veh_arrives_facilty_event2,
        veh2_arrives_facilty_event2,
        person_leaves_veh_event,
        person_enters_veh2_event,
        veh_arrives_facilty_event3,
        veh2_arrives_facilty_event3
    ]
    for elem in events:
        handler.process_event(elem)
    
    handler.finalise()

    gdf = handler.result_dfs["vehicle_stop_to_stop_passenger_counts_bus"]
    assert set(gdf['veh_id'].unique()) == {'bus1','bus2'}

    cols = list(range(handler.config.time_periods))
    for c in cols:
        assert c in gdf.columns
        assert 'total' in gdf.columns
    df = gdf.loc[:, cols]
    assert np.sum(df.values) == 4 / handler.config.scale_factor
    assert np.sum(df.values) == gdf.total.sum()
    if 'subpopulation' in gdf.columns:
        assert set(gdf.loc[:, 'subpopulation']) == {'poor', 'rich', np.nan}


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
    gdf = handler.result_dfs["stop_to_stop_passenger_counts_bus_subpopulation"]
    cols = list(range(handler.config.time_periods))
    for c in cols:
        assert c in gdf.columns
    df = gdf.loc[:, cols]
    assert np.sum(df.values) == 3 / handler.config.scale_factor
    if 'subpopulation' in gdf.columns:
        assert set(gdf.loc[:, 'subpopulation']) == {'poor', 'rich', np.nan}

    gdf = handler.result_dfs["stop_to_stop_passenger_counts_bus"]
    cols = list(range(handler.config.time_periods))
    for c in cols:
        assert c in gdf.columns
        assert 'total' in gdf.columns
    df = gdf.loc[:, cols]
    assert np.sum(df.values) == 3 / handler.config.scale_factor
    assert np.sum(df.values) == gdf.total.sum()
    if 'subpopulation' in gdf.columns:
        assert set(gdf.loc[:, 'subpopulation']) == {'poor', 'rich', np.nan}


# simple case no attributes
@pytest.fixture
def bus_stop_to_stop_handler_simple(test_config, input_manager):
    handler = event_handlers.StopToStopPassengerCounts(test_config, mode='bus', groupby_person_attribute=None)
    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    return handler

@pytest.fixture
def bus_vehicle_stop_to_stop_handler_simple(test_config, input_manager):
    handler = event_handlers.VehicleStopToStopPassengerCounts(test_config, mode='bus', groupby_person_attribute=None)
    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    return handler

@pytest.fixture
def bus_vehicle_passenger_log_handler_simple(test_config, input_manager):
    handler = event_handlers.VehiclePassengerLog(test_config, mode='bus', groupby_person_attribute=None)
    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    return handler


def test_stop_to_stop_finalise_bus_simple(
        bus_stop_to_stop_handler_simple,
        veh_arrives_facilty_event1,
        person_enters_veh_event,
        person2_enters_veh_event,
        veh_arrives_facilty_event2,
        person_leaves_veh_event,
        veh_arrives_facilty_event3
):
    handler = bus_stop_to_stop_handler_simple
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
    assert len(handler.result_dfs) == 1
    gdf = handler.result_dfs["stop_to_stop_passenger_counts_bus"]
    cols = list(range(handler.config.time_periods))
    for c in cols:
        assert c in gdf.columns
        assert 'total' in gdf.columns
    df = gdf.loc[:, cols]
    assert np.sum(df.values) == 3 / handler.config.scale_factor
    assert np.sum(df.values) == gdf.total.sum()

# Vehicle departs stop test
def test_vehicle_departs_facility(test_config, input_manager):
    handler = event_handlers.VehicleDepartureLog(test_config, mode = "all")
    handler.build(input_manager.resources)
            
    for elem in handler.resources['events'].elems:
        handler.process_event(elem)

    log_length = len(handler.vehicle_departure_log.chunk)
    chunk_four = handler.vehicle_departure_log.chunk[3]

    assert log_length == 8
    assert chunk_four == {
        'veh_id': 'bus2',
        'veh_mode': 'bus',
        'veh_route': 'work_bound',
        'stop_id': 'work_stop_in',
        'departure_time': 31363,
        'delay': -137
    }

# Vehicle link speeds test using full events/config
def test_link_vehicle_speeds_events_file(test_config, input_manager):
    handler = event_handlers.LinkVehicleSpeeds(test_config, mode="all")
    handler.build(input_manager.resources)

    for elem in handler.resources['events'].elems:
        handler.process_event(elem)

    handler.finalise()

    # test that the minimum speed is realistic (<10 kph)
    data_cols = [i for i in range(0, test_config.time_periods)]

    for  df in handler.result_dfs.values():
        df = df[data_cols]
        df.replace(0, np.nan, inplace=True)
        min_speed = df.min().min() # min value across all time columns
        assert min_speed >= 10

# Vehicle link logs test
def test_vehicle_link_log(test_config, input_manager):
    handler = event_handlers.VehicleLinkLog(test_config, mode="all")
    handler.build(input_manager.resources)
            
    for elem in handler.resources['events'].elems:
        handler.process_event(elem)

    log_length = len(handler.vehicle_link_log.chunk)
    chunk_five = handler.vehicle_link_log.chunk[4]

    assert log_length == 10
    assert chunk_five == {
        'veh_id': 'chris',
        'veh_mode': 'car',
        'link_id': '4-3',
        'entry_time': 59401,
        'exit_time': 59406
    } 

# Vehicle passenger boardings/alightings test
def test_vehicle_passenger_boarding(test_config, input_manager):
    handler = event_handlers.VehiclePassengerLog(test_config, mode = "all")
    handler.build(input_manager.resources)

    for elem in handler.resources['events'].elems:
        handler.process_event(elem)

    # there are 8 boardings (and 8 alightings) in the test outputs 
    n_boardings = 0
    n_alightings = 0
    for i in handler.vehicle_passenger_log.chunk:
        if i.get('event_type') == 'PersonEntersVehicle':
            n_boardings += 1
        elif i.get('event_type') == 'PersonLeavesVehicle':
            n_alightings += 1

    assert n_boardings == 8
    assert n_alightings == 8

    assert handler.vehicle_passenger_log.chunk[6] == {
        'agent_id': 'fred', 
        'event_type': 'PersonEntersVehicle', 
        'veh_id': 'bus2', 
        'stop_id': 'home_stop_out', 
        'time': '30601.0', 
        'veh_mode': 'bus', 
        'veh_route': 'work_bound'
    }

# Agent Tolls Test
@pytest.fixture
def person_toll_events():
    """
    Inlcudes a PT Driver incurring a toll
    These should be excluded from results
    """
    string = """
        <events>
            <event time="200.0" type="personMoney" person="fred" amount="-5" purpose="toll"/>
            <event time="300.0" type="personMoney" person="fred" amount="-10" purpose="toll"/>
            <event time="400.0" type="personMoney" person="chris" amount="-1" purpose="toll"/>
            <event time="500.0" type="personMoney" person="pt_bus1_bus" amount="-1" purpose="toll"/>
        </events>
        """

    return etree.fromstring(string)

def test_agent_tolls_process_event(test_config, person_toll_events, input_manager):
    handler = event_handlers.AgentTollsLog(test_config)
    events = person_toll_events
    resources = input_manager.resources
    handler.build(resources)
    for elem in events:
        handler.process_event(elem)

    target = {
        'fred': {'toll_total': 15, 'tolls_incurred': 2},
        'chris': {'toll_total': 1, 'tolls_incurred': 1}
    }

    assert handler.toll_log_summary == target

def test_agent_tolls_process_event_with_subpopulation(test_config, person_toll_events, input_manager):
    """
    Agent attributes supplied via input_manager/test_config see file:
    ./tests/test_fixtures/output_personAttributes.xml.gz
    """
    handler = event_handlers.AgentTollsLog(test_config, groupby_person_attribute="subpopulation")
    resources = input_manager.resources
    handler.build(resources)

    events = person_toll_events

    for elem in events:
        handler.process_event(elem)

    target = {
        'fred': {'toll_total': 15, 'tolls_incurred': 2, 'class': 'poor'},
        'chris': {'toll_total': 1, 'tolls_incurred': 1, 'class': 'rich'}
    }
    
    assert handler.toll_log_summary == target

def test_agent_tolls_chunkwriter(test_config, person_toll_events, input_manager):
    handler = event_handlers.AgentTollsLog(test_config)
    resources = input_manager.resources
    handler.build(resources)

    events = person_toll_events
        
    for elem in events:
        handler.process_event(elem)

    target_chunk = [
        {'agent_id': 'fred', 'toll_amount': 5, 'time': 200},
        {'agent_id': 'fred', 'toll_amount': 10, 'time': 300},
        {'agent_id': 'chris', 'toll_amount': 1, 'time': 400}
    ]

    assert handler.agent_tolls_log.chunk == target_chunk

def test_agent_tolls_finalise(test_config, person_toll_events, input_manager):
    """
    tests pandas operations are working as expected
    """
    handler = event_handlers.AgentTollsLog(test_config, groupby_person_attribute="subpopulation")
    resources = input_manager.resources
    handler.build(resources)

    events = person_toll_events

    for elem in events:
        handler.process_event(elem)

    handler.finalise()

    target = {
        'fred': {'toll_total': 15, 'tolls_incurred': 2, 'class': 'poor'},
        'chris': {'toll_total': 1, 'tolls_incurred': 1, 'class': 'rich'}
    }

    target_grouped = {
        'poor': {'toll_total': 15, 'avg_per_agent': 15, 'tolled_agents': 1, 'tolls_incurred': 2},
        'rich': {'toll_total': 1, 'avg_per_agent': 1, 'tolled_agents': 1, 'tolls_incurred': 1}
    }
    
    results = handler.result_dfs[handler.name + '_summary'].to_dict(orient = 'index')
    results_grouped = handler.result_dfs[handler.name + '_summary_subpopulation'].to_dict(
        orient = 'index'
    )
    
    assert results == target
    assert results_grouped == target_grouped


# Event Handler Manager Load all tools with car mode
def test_load_all_event_handler_manager_with_mode_car(test_config, test_paths):
    input_workstation = inputs.InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()
    event_workstation = EventHandlerWorkStation(test_config)
    event_workstation.connect(managers=None, suppliers=[input_workstation])
    event_workstation.load_all_tools(mode='car')
    event_workstation.build(write_path=test_outputs)

    for handler_name, handler in event_workstation.resources.items():
        for name, gdf in handler.result_dfs.items():
            if 'agent_tolls_log' not in name:  # handler does not conform to test criteria
                cols = list(range(handler.config.time_periods))
                for c in cols:
                    assert c in gdf.columns
                df = gdf.loc[:, cols]
                assert np.sum(df.values)


# Event Handler Manager Load all tools with bus mode
def test_load_all_event_handler_manager_with_mode_bus(test_config, test_paths):
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
            if 'agent_tolls_log' not in name:  # handler does not conform to test criteria
                cols = list(range(handler.config.time_periods))
                for c in cols:
                    assert c in gdf.columns
                df = gdf.loc[:, cols]
                assert np.sum(df.values)


# Event Handler Manager Load all tools with car mode and groupby subpopulation
def test_load_all_event_handler_manager_with_mode_car_and_groupby_subpopulation(test_config, test_paths):
    input_workstation = inputs.InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()
    event_workstation = EventHandlerWorkStation(test_config)
    event_workstation.connect(managers=None, suppliers=[input_workstation])
    event_workstation.load_all_tools(mode='car', groupby_person_attribute="subpopulation")
    event_workstation.build(write_path=test_outputs)

    for handler_name, handler in event_workstation.resources.items():
        for name, gdf in handler.result_dfs.items():
            if 'agent_tolls_log' not in name:  # handler does not conform to test criteria
                cols = list(range(handler.config.time_periods))
                for c in cols:
                    assert c in gdf.columns
                df = gdf.loc[:, cols]
                assert np.sum(df.values)