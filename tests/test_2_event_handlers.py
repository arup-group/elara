import sys
import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import inputs, handlers
from elara.handlers.network_event_handlers import *
sys.path.append(os.path.abspath('../tests'))


test_floor_data = [
    (0, 0),
    (1*3600, 1),
    (24*3600 - 1, 23),
    (24*3600, 0)
]


@pytest.mark.parametrize("seconds,hour", test_floor_data)
def test_table_position_floor(seconds, hour):
    assert handlers.network_event_handlers.table_position(
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


@pytest.fixture
def config():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    return Config(config_path)


@pytest.fixture
def network(config):
    return inputs.Network(config.network_path, config.crs)


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


# Base
@pytest.fixture
def base_handler(network, transit_vehicles, transit_schedule, attributes):
    base_handler = handlers.network_event_handlers.Handler(
        network,
        transit_schedule,
        transit_vehicles,
        attributes,
        'car',
        24,
        0.01
    )
    assert base_handler.mode == 'car'
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
    assert base_handler.vehicle_mode('bus1') == 'bus'
    assert base_handler.vehicle_mode('not_a_transit_vehicle') == "car"


def test_empty_rows(base_handler, test_df):
    assert len(base_handler.remove_empty_rows(test_df)) == 2


@pytest.fixture
def events(config):
    return inputs.Events(config.events_path).elems


@pytest.fixture
def car_enters_link_event():
    time = 6.5 * 60 * 60
    string = """
        <event time = "23400.0" type = "vehicle enters traffic" person = "chris"
        link = "1-2" vehicle = "chris" networkMode = "car" relativePosition = "1.0" />
        """
    return etree.fromstring(string)


@pytest.fixture
def bus_enters_link_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23400.0" type="entered link" vehicle="bus1" link="1-2" />
        """
    return etree.fromstring(string)


# Volume Counts
# Car
@pytest.fixture
def test_car_volume_count_handler(network, transit_vehicles, transit_schedule, attributes):
    periods = 24
    handler = VolumeCounts(
        network,
        transit_schedule,
        transit_vehicles,
        attributes,
        'car',
        periods,
        0.01
    )
    assert 'not_applicable' in handler.classes
    assert len(handler.classes) == len(attributes.classes)
    assert list(handler.class_indices.keys()) == handler.classes
    assert len(handler.elem_ids) == len(network.link_gdf)
    assert list(handler.elem_indices.keys()) == handler.elem_ids
    assert handler.counts.shape == (len(network.link_gdf), len(attributes.classes), periods)
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
    for name, gdf in handler.result_gdfs.items():
        cols = list(range(handler.periods))
        for c in cols:
            assert c in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 14 / handler.scale_factor
        if 'class' in gdf.columns:
            assert set(gdf.loc[:, 'class']) == set(handler.attributes.classes)


# Bus
@pytest.fixture
def test_bus_volume_count_handler(network, transit_vehicles, transit_schedule, attributes):
    periods = 24
    handler = VolumeCounts(
        network,
        transit_schedule,
        transit_vehicles,
        attributes,
        'bus',
        periods,
        0.01
    )
    assert 'not_applicable' in handler.classes
    assert len(handler.classes) == len(attributes.classes)
    assert list(handler.class_indices.keys()) == handler.classes
    assert handler.counts.shape == (len(network.link_gdf), len(attributes.classes), periods)
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
    handler.finalise()
    for name, gdf in handler.result_gdfs.items():
        cols = list(range(handler.periods))
        for c in cols:
            assert c in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 12 / handler.scale_factor
        if 'class' in gdf.columns:
            assert set(gdf.loc[:, 'class']) == set(handler.attributes.classes)


# Passenger Counts Handler Tests

@pytest.fixture
def test_bus_passenger_count_handler(network, transit_vehicles, transit_schedule, attributes):
    periods = 24
    handler = PassengerCounts(
        network,
        transit_schedule,
        transit_vehicles,
        attributes,
        'bus',
        periods,
        0.01
    )
    assert 'not_applicable' in handler.classes
    assert len(handler.classes) == len(attributes.classes)
    assert list(handler.class_indices.keys()) == handler.classes
    assert handler.counts.shape == (len(network.link_gdf), len(attributes.classes), periods)
    return handler


@pytest.fixture
def driver_enters_veh_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23400.0" type="PersonEntersVehicle" person="pt_bus1_Bus" vehicle="bus1" />
        """
    return etree.fromstring(string)


@pytest.fixture
def person_enters_veh_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23401.0" type="PersonEntersVehicle" person="gerry" vehicle="bus1"  />
        """
    return etree.fromstring(string)


@pytest.fixture
def person2_enters_veh_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23401.0" type="PersonEntersVehicle" person="chris" vehicle="bus1"  />
        """
    return etree.fromstring(string)


@pytest.fixture
def bus_leaves_link_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23405.0" type="left link" vehicle="bus1" link="2-3"  />
        """
    return etree.fromstring(string)


@pytest.fixture
def person_leaves_veh_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23410.0" type="PersonLeavesVehicle" person="gerry" vehicle="bus1" />
        """
    return etree.fromstring(string)


@pytest.fixture
def person2_leaves_veh_event():
    time = 6.5 * 60 * 60
    string = """
        <event time="23410.0" type="PersonLeavesVehicle" person="chris" vehicle="bus1" />
        """
    return etree.fromstring(string)


def test_passenger_count_process_single_event_driver(
        test_bus_passenger_count_handler,
        driver_enters_veh_event):
    handler = test_bus_passenger_count_handler
    elem = driver_enters_veh_event
    handler.process_event(elem)
    assert sum(handler.veh_occupancy.values()) == 0
    assert np.sum(handler.counts) == 0


def test_passenger_count_process_single_event_link(
        test_bus_passenger_count_handler,
        bus_leaves_link_event):
    handler = test_bus_passenger_count_handler
    elem = bus_leaves_link_event
    handler.process_event(elem)
    assert sum(handler.veh_occupancy.values()) == 0
    assert np.sum(handler.counts) == 0


def test_passenger_count_process_single_event_passenger(
        test_bus_passenger_count_handler,
        person_enters_veh_event):
    handler = test_bus_passenger_count_handler
    elem = person_enters_veh_event
    handler.process_event(elem)
    assert sum([v for vo in handler.veh_occupancy.values() for v in vo.values()]) == 1
    assert np.sum(handler.counts) == 0


def test_passenger_count_process_events(
        test_bus_passenger_count_handler,
        person_enters_veh_event,
        person2_enters_veh_event,
        car_enters_link_event,
        bus_leaves_link_event):
    handler = test_bus_passenger_count_handler
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
        test_bus_passenger_count_handler,
        events
):
    handler = test_bus_passenger_count_handler
    for elem in events:
        handler.process_event(elem)
    handler.finalise()
    for name, gdf in handler.result_gdfs.items():
        cols = list(range(handler.periods))
        for c in cols:
            assert c in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 8 / handler.scale_factor
        if 'class' in gdf.columns:
            assert set(gdf.loc[:, 'class']) == set(handler.attributes.classes)


# Stop Interactions

@pytest.fixture
def test_bus_passenger_interaction_handler(network, transit_vehicles, transit_schedule, attributes):
    periods = 24
    handler = StopInteractions(
        network,
        transit_schedule,
        transit_vehicles,
        attributes,
        'bus',
        periods,
        0.01
    )
    assert 'not_applicable' in handler.classes
    assert len(handler.classes) == len(attributes.classes)
    assert list(handler.class_indices.keys()) == handler.classes
    assert handler.boardings.shape == (
        len(transit_schedule.stop_gdf),
        len(attributes.classes),
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
    for name, gdf in handler.result_gdfs.items():
        cols = list(range(handler.periods))
        for c in cols:
            assert c in gdf.columns
        df = gdf.loc[:, cols]
        assert np.sum(df.values) == 4 / handler.scale_factor
        if 'class' in gdf.columns:
            assert set(gdf.loc[:, 'class']) == set(handler.attributes.classes)

