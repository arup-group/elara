import sys
import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree
from datetime import datetime, timedelta


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara import plan_handlers
from elara.plan_handlers import PlanHandlerWorkStation

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
test_inputs = os.path.join(test_dir, "test_intermediate_data")
test_outputs = os.path.join(test_dir, "test_outputs")
if not os.path.exists(test_outputs):
    os.mkdir(test_outputs)
benchmarks_path = os.path.join(test_outputs, "benchmarks")
if not os.path.exists(benchmarks_path):
    os.mkdir(benchmarks_path)


test_matsim_time_data = [
    ('00:00:00', 0),
    ('01:01:01', 3661),
    (None, None),
]


@pytest.mark.parametrize("time,seconds", test_matsim_time_data)
def test_convert_time(time, seconds):
    assert plan_handlers.convert_time_to_seconds(time) == seconds


# Config
@pytest.fixture
def test_config():
    config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
    config = Config(config_path)
    assert config
    return config


@pytest.fixture
def test_config_v12():
    config_path = os.path.join(test_dir, 'test_xml_scenario_v12.toml')
    config = Config(config_path)
    assert config
    return config


def test_v12_config(test_config_v12):
    assert test_config_v12.version == 12


# Paths
@pytest.fixture
def test_paths(test_config):
    paths = PathFinderWorkStation(test_config)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths


@pytest.fixture
def test_paths_v12(test_config_v12):
    paths = PathFinderWorkStation(test_config_v12)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths


# Input Manager
@pytest.fixture
def input_manager(test_config, test_paths):
    input_workstation = InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()
    return input_workstation


@pytest.fixture
def input_manager_v12(test_config_v12, test_paths_v12):
    input_workstation = InputsWorkStation(test_config_v12)
    input_workstation.connect(managers=None, suppliers=[test_paths_v12])
    input_workstation.load_all_tools()
    input_workstation.build()
    return input_workstation


# Base
@pytest.fixture
def base_handler(test_config, input_manager):
    base_handler = plan_handlers.PlanHandlerTool(test_config, 'all')
    assert base_handler.option == 'all'
    base_handler.build(input_manager.resources, write_path=test_outputs)
    return base_handler


@pytest.fixture
def base_handler_v12(test_config_v12, input_manager):
    base_handler = plan_handlers.PlanHandlerTool(test_config_v12, 'all')
    assert base_handler.option == 'all'
    base_handler.build(input_manager.resources, write_path=test_outputs)
    return base_handler


mode_distances = [
    ({'a':2, 'b':0.5}, 'a'),
    ({'a':2, 'b':2}, 'a'),
    ({'a':2}, 'a'),
    ({'a':2, 'b':-1}, 'a'),
    ({'transit_walk':2, 'b':1}, 'b'),
    ({None: 1}, None),
]


@pytest.mark.parametrize("modes,mode", mode_distances)
def get_furthest_mode(modes, mode):
    assert plan_handlers.PlanHandlerTool.get_furthest_mode(modes) == mode


def test_extract_mode_from_v11_route_elem(base_handler):
    class Resource:
        route_to_mode_map = {"a":"bus"}
    base_handler.resources['transit_schedule'] = Resource()
    string = """
    <route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" 
    distance="10100.0">PT1===home_stop_out===city_line===a===work_stop_in</route>
    """
    elem = etree.fromstring(string)
    assert base_handler.extract_mode_from_v11_route_elem(elem) == "bus"


def test_extract_routeid_from_v12_route_elem(base_handler):
    class Resource:
        route_to_mode_map = {"a":"bus"}
    base_handler.resources['transit_schedule'] = Resource()
    string = """
    <route type="default_pt" start_link="1" 
    end_link="2" trav_time="00:33:03" distance="2772.854305426653">
    {"transitRouteId":"a","boardingTime":"09:15:00","transitLineId":"b",
    "accessFacilityId":"1","egressFacilityId":"2"}</route>
    """
    elem = etree.fromstring(string)
    assert base_handler.extract_routeid_from_v12_route_elem(elem) == "a"


def test_extract_mode_from_route_elem_v11(base_handler):
    class Resource:
        route_to_mode_map = {"a":"bus"}
    base_handler.resources['transit_schedule'] = Resource()
    string = """
    <route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" 
    distance="10100.0">PT1===home_stop_out===city_line===a===work_stop_in</route>
    """
    elem = etree.fromstring(string)
    assert base_handler.extract_mode_from_route_elem("pt", elem) == "bus"


def test_extract_mode_from_route_elem_v12(base_handler_v12):
    class Resource:
        route_to_mode_map = {"a":"bus"}
    base_handler_v12.resources['transit_schedule'] = Resource()
    print(base_handler_v12.config.version)
    string = """
    <route type="default_pt" start_link="1" 
    end_link="2" trav_time="00:33:03" distance="2772.854305426653">
    {"transitRouteId":"a","boardingTime":"09:15:00","transitLineId":"b",
    "accessFacilityId":"1","egressFacilityId":"2"}</route>
    """
    elem = etree.fromstring(string)
    assert base_handler_v12.extract_mode_from_route_elem("bus", elem) == "bus"


### Utility Handler ###

@pytest.fixture
def person_single_plan_elem():
    string = """
        <person id="test1">
		<plan score="10" selected="yes">
		</plan>
	</person>
        """
    return etree.fromstring(string)


@pytest.fixture
def person_plans_elem():
    string = """
        <person id="test2">
		<plan score="10" selected="yes">
		</plan>
        <plan score="8" selected="no">
		</plan>
	</person>
        """
    return etree.fromstring(string)


@pytest.fixture
def utility_handler(test_config, input_manager):
    handler = plan_handlers.UtilityLogs(test_config, 'all')
    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    assert len(handler.utility_log.chunk) == 0
    return handler


def test_utility_handler_process_single_plan(utility_handler, person_single_plan_elem):
    assert len(utility_handler.utility_log) == 0
    utility_handler.process_plans(person_single_plan_elem)
    assert len(utility_handler.utility_log) == 1
    assert utility_handler.utility_log.chunk == [{'agent': 'test1','score': '10'}]


def test_utility_handler_process_multi_plan(utility_handler, person_plans_elem):
    assert len(utility_handler.utility_log) == 0
    utility_handler.process_plans(person_plans_elem)
    assert len(utility_handler.utility_log) == 1
    assert utility_handler.utility_log.chunk == [{'agent': 'test2','score': '10'}]


def test_utility_handler_process_plans(utility_handler, person_single_plan_elem, person_plans_elem):
    assert len(utility_handler.utility_log) == 0
    utility_handler.process_plans(person_single_plan_elem)
    utility_handler.process_plans(person_plans_elem)
    assert len(utility_handler.utility_log) == 2
    assert utility_handler.utility_log.chunk == [{'agent': 'test1','score': '10'}, {'agent': 'test2','score': '10'}]


### Leg Log Handler ###
# Wrapping
test_matsim_time_data = [
    (['06:00:00', '12:45:00', '18:30:00'], '1-18:30:00'),
    (['06:00:00', '12:45:00', '24:00:00'], '2-00:00:00'),
    (['06:00:00', '24:00:00', '08:30:00'], '2-08:30:00'),
    (['06:00:00', '18:45:00', '12:30:00'], '2-12:30:00'),
    (['06:00:00', '18:45:00', '18:45:00'], '1-18:45:00'),
    (['00:00:00', '12:45:00', '18:45:00'], '1-18:45:00'),
    (['06:00:00', '04:45:00', '02:45:00'], '3-02:45:00'),
    (['00:00:00'], '1-00:00:00'),
    (['24:00:00'], '2-00:00:00'),
]

non_wrapping_test_matsim_time_data = [
    (['06:00:00', '12:45:00', '18:30:00'], '1-18:30:00'),
    (['06:00:00', '12:45:00', '24:00:00'], '2-00:00:00'),
    (['06:00:00', '24:00:00', '08:30:00'], '1-08:30:00'),
    (['06:00:00', '18:45:00', '12:30:00'], '1-12:30:00'),
    (['00:00:00'], '1-00:00:00'),
    (['24:00:00'], '2-00:00:00'),
]

@pytest.mark.parametrize("times,final_string", non_wrapping_test_matsim_time_data)
def test_matsim_time_to_datetime(times, final_string):
    current_dt = None
    for new_time_str in times:
        current_dt = plan_handlers.matsim_time_to_datetime(
            current_dt,
            new_time_str,
            base_year=1900,
            base_month=1,
            )
    assert isinstance(current_dt, datetime)
    assert current_dt == datetime.strptime(f"{final_string}", '%d-%H:%M:%S')


test_durations_data = [
    (
        None, datetime(year=2020, month=4, day=1, hour=0),
        timedelta(hours=0), datetime(year=2020, month=4, day=1, hour=0)
    ),
    (
        None, datetime(year=2020, month=4, day=1, hour=1),
        timedelta(hours=1), datetime(year=2020, month=4, day=1, hour=0)
    ),
    (
        datetime(year=2020, month=4, day=1, hour=1), datetime(year=2020, month=4, day=1, hour=1),
        timedelta(hours=0), datetime(year=2020, month=4, day=1, hour=1)
    ),
    (
        datetime(year=2020, month=4, day=1, hour=1), datetime(year=2020, month=4, day=1, hour=2),
        timedelta(hours=1), datetime(year=2020, month=4, day=1, hour=1)
    ),
    (
        datetime(year=2020, month=4, day=1, hour=2), datetime(year=2020, month=4, day=1, hour=1),
        timedelta(hours=-1), datetime(year=2020, month=4, day=1, hour=2)
    ),
]
@pytest.mark.parametrize("start,end,duration,start_time", test_durations_data)
def test_safe_duration(start, end, duration, start_time):
    d, st = plan_handlers.safe_duration(start, end)
    assert d == duration
    assert st == start_time


test_distance_data = [
    (0,0,0,0,0),
    (1,1,1,1,0),
    (0,0,3,4,5),
    (3,4,0,0,5),
    (3,0,0,-4,5)
]
@pytest.mark.parametrize("x1,y1,x2,y2,dist", test_distance_data)
def test_distance(x1,y1,x2,y2,dist):
    assert plan_handlers.distance(x1,y1,x2,y2) == dist


# Normal Case
@pytest.fixture
def agent_leg_log_handler(test_config, input_manager):
    handler = plan_handlers.LegLogs(test_config, 'all')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.activities_log.chunk) == 0
    assert len(handler.legs_log.chunk) == 0

    return handler


def test_agent_log_handler(agent_leg_log_handler):
    handler = agent_leg_log_handler

    plans = handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert len(handler.activities_log.chunk) == 23
    assert len(handler.legs_log.chunk) == 18


@pytest.fixture
def agent_leg_log_handler_finalised(agent_leg_log_handler):
    handler = agent_leg_log_handler
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    handler.finalise()
    return handler


def test_finalised_logs(agent_leg_log_handler_finalised):
    handler = agent_leg_log_handler_finalised

    assert len(handler.results) == 0


# Plans Wrapping case

# Bad Config (plans wrap past 24hrs)
@pytest.fixture
def test_bad_plans_config():
    config_path = os.path.join(test_dir, 'test_xml_scenario_bad_plans.toml')
    config = Config(config_path)
    assert config
    return config


# Paths
@pytest.fixture
def test_bad_plans_paths(test_bad_plans_config):
    paths = PathFinderWorkStation(test_bad_plans_config)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths


# Input Manager
@pytest.fixture
def input_bad_plans_manager(test_bad_plans_config, test_bad_plans_paths):
    input_workstation = InputsWorkStation(test_bad_plans_config)
    input_workstation.connect(managers=None, suppliers=[test_bad_plans_paths])
    input_workstation.load_all_tools()
    input_workstation.build()
    return input_workstation


@pytest.fixture
def agent_leg_log_handler_bad_plans(test_bad_plans_config, input_bad_plans_manager):
    handler = plan_handlers.LegLogs(test_bad_plans_config, 'all')

    resources = input_bad_plans_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.activities_log.chunk) == 0
    assert len(handler.legs_log.chunk) == 0

    return handler


@pytest.fixture
def agent_leg_log_handler_finalised_bad_plans(agent_leg_log_handler_bad_plans):
    handler = agent_leg_log_handler_bad_plans
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    assert handler.activities_log.chunk[-1].get('end_day') == 1
    assert handler.legs_log.chunk[-1].get('end_day') == 1
    handler.finalise()
    return handler


def test_finalised_logs_bad_plans(agent_leg_log_handler_finalised_bad_plans):
    handler = agent_leg_log_handler_finalised_bad_plans
    assert len(handler.results) == 0


### Trip Log Handler ###

# Normal Case
@pytest.fixture
def agent_trip_handler(test_config, input_manager):
    handler = plan_handlers.TripLogs(test_config, 'all')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.activities_log.chunk) == 0
    assert len(handler.trips_log.chunk) == 0

    return handler


def test_agent_trip_log_process_person(agent_trip_handler):
    handler = agent_trip_handler

    person = """
    <person id="nick">
        <plan score="1" selected="yes">
            <activity type="home" link="1-2" x="0.0" y="0.0" end_time="08:00:00" >
            </activity>
            <leg mode="car" dep_time="08:00:00" trav_time="00:00:04">
            <route type="links" start_link="1-2" end_link="1-5" trav_time="00:00:04" distance="10100.0">1-2 2-1 1-5</route>
            </leg>
            <activity type="work" link="1-5" x="0.0" y="10000.0" end_time="17:30:00" >
            </activity>
        </plan>
    </person>
    """
    person = etree.fromstring(person)
    handler.process_plans(person)

    assert handler.activities_log.chunk[0]["start_s"] == 0
    assert handler.activities_log.chunk[0]["duration_s"] == 8*60*60
    assert handler.activities_log.chunk[0]["end_s"] == 8*60*60
    assert handler.activities_log.chunk[0]["act"] == "home"

    assert handler.trips_log.chunk[0]["start_s"] == 8*60*60
    assert handler.trips_log.chunk[0]["duration_s"] == 4
    assert handler.trips_log.chunk[0]["end_s"] == 8*60*60 + 4
    assert handler.trips_log.chunk[0]["mode"] == "car"

    assert handler.activities_log.chunk[1]["start_s"] == 8*60*60 + 4
    assert handler.activities_log.chunk[1]["duration_s"] == 17.5*60*60 - (8*60*60 + 4)
    assert handler.activities_log.chunk[1]["end_s"] == 17.5*60*60
    assert handler.activities_log.chunk[1]["act"] == "work"


def test_agent_trip_log_process_pt_bus_person(agent_trip_handler):
    handler = agent_trip_handler
    handler.resources['transit_schedule'].route_to_mode_map["rail_dummy"] = "rail"

    person = """
    <person id="nick">
        <plan score="1" selected="yes">
            <activity type="home" link="1-2" x="0.0" y="0.0" end_time="08:00:00" >
            </activity>
            <leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="1-2" end_link="1-2" trav_time="00:01:18" distance="65.0"></route>
			</leg>
			<activity type="pt interaction" link="1-2" x="50.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="pt" trav_time="00:43:42">
				<route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" distance="10100.0">PT1===home_stop_out===city_line===work_bound===work_stop_in</route>
			</leg>
			<activity type="pt interaction" link="3-4" x="10050.0" y="0.0" max_dur="00:00:00" >
			</activity>
            <leg mode="pt" trav_time="00:43:42">
				<route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" distance="1010.0">PT1===home_stop_out===rail_dummy===rail_dummy===work_stop_in</route>
			</leg>
			<activity type="pt interaction" link="3-4" x="10050.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="3-4" end_link="3-4" trav_time="00:01:18" distance="65.0"></route>
			</leg>
            <activity type="work" link="1-5" x="0.0" y="10000.0" end_time="17:30:00" >
            </activity>
        </plan>
    </person>
    """
    person = etree.fromstring(person)
    handler.process_plans(person)

    assert handler.activities_log.chunk[0]["start_s"] == 0
    assert handler.activities_log.chunk[0]["duration_s"] == 8*60*60
    assert handler.activities_log.chunk[0]["end_s"] == 8*60*60
    assert handler.activities_log.chunk[0]["act"] == "home"

    assert handler.trips_log.chunk[0]["start_s"] == 8*60*60
    assert handler.trips_log.chunk[0]["duration_s"] == 1.5*60*60
    assert handler.trips_log.chunk[0]["end_s"] == 8*60*60 + 1.5*60*60
    assert handler.trips_log.chunk[0]["mode"] == "bus"

    assert handler.activities_log.chunk[1]["start_s"] == 8*60*60 + 1.5*60*60
    assert handler.activities_log.chunk[1]["duration_s"] == 17.5*60*60 - (8*60*60 + 1.5*60*60)
    assert handler.activities_log.chunk[1]["end_s"] == 17.5*60*60
    assert handler.activities_log.chunk[1]["act"] == "work"


def test_agent_trip_log_process_pt_rail_person(agent_trip_handler):
    handler = agent_trip_handler
    handler.resources['transit_schedule'].route_to_mode_map["rail_dummy"] = "rail"

    person = """
    <person id="nick">
        <plan score="1" selected="yes">
            <activity type="home" link="1-2" x="0.0" y="0.0" end_time="08:00:00" >
            </activity>
            <leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="1-2" end_link="1-2" trav_time="00:01:18" distance="65.0"></route>
			</leg>
			<activity type="pt interaction" link="1-2" x="50.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="pt" trav_time="00:43:42">
				<route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" distance="10100.0">PT1===home_stop_out===city_line===work_bound===work_stop_in</route>
			</leg>
			<activity type="pt interaction" link="3-4" x="10050.0" y="0.0" max_dur="00:00:00" >
			</activity>
            <leg mode="pt" trav_time="00:43:42">
				<route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" distance="10101.0">PT1===home_stop_out===rail_dummy===rail_dummy===work_stop_in</route>
			</leg>
			<activity type="pt interaction" link="3-4" x="10050.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="3-4" end_link="3-4" trav_time="00:01:18" distance="65.0"></route>
			</leg>
            <activity type="work" link="1-5" x="0.0" y="10000.0" end_time="17:30:00" >
            </activity>
        </plan>
    </person>
    """
    person = etree.fromstring(person)
    handler.process_plans(person)

    assert handler.activities_log.chunk[0]["start_s"] == 0
    assert handler.activities_log.chunk[0]["duration_s"] == 8*60*60
    assert handler.activities_log.chunk[0]["end_s"] == 8*60*60
    assert handler.activities_log.chunk[0]["act"] == "home"

    assert handler.trips_log.chunk[0]["start_s"] == 8*60*60
    assert handler.trips_log.chunk[0]["duration_s"] == 1.5*60*60
    assert handler.trips_log.chunk[0]["end_s"] == 8*60*60 + 1.5*60*60
    assert handler.trips_log.chunk[0]["mode"] == "rail"

    assert handler.activities_log.chunk[1]["start_s"] == 8*60*60 + 1.5*60*60
    assert handler.activities_log.chunk[1]["duration_s"] == 17.5*60*60 - (8*60*60 + 1.5*60*60)
    assert handler.activities_log.chunk[1]["end_s"] == 17.5*60*60
    assert handler.activities_log.chunk[1]["act"] == "work"


def test_agent_trip_log_handler(agent_trip_handler):
    handler = agent_trip_handler

    plans = handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert len(handler.activities_log.chunk) == 15
    assert len(handler.trips_log.chunk) == 10


@pytest.fixture
def agent_trip_log_handler_finalised(agent_trip_handler):
    handler = agent_trip_handler
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    handler.finalise()
    return handler


def test_finalised_logs(agent_trip_log_handler_finalised):
    handler = agent_trip_log_handler_finalised

    assert len(handler.results) == 0


# Plans Wrapping case

@pytest.fixture
def agent_trip_log_handler_bad_plans(test_bad_plans_config, input_bad_plans_manager):
    handler = plan_handlers.TripLogs(test_bad_plans_config, 'all')

    resources = input_bad_plans_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.activities_log.chunk) == 0
    assert len(handler.trips_log.chunk) == 0

    return handler


@pytest.fixture
def agent_trips_log_handler_finalised_bad_plans(agent_trip_log_handler_bad_plans):
    handler = agent_trip_log_handler_bad_plans
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    assert handler.activities_log.chunk[-1].get('end_day') == 1
    assert handler.trips_log.chunk[-1].get('end_day') == 1
    handler.finalise()
    return handler


def test_finalised_trips_logs_bad_plans(agent_trips_log_handler_finalised_bad_plans):
    handler = agent_trips_log_handler_finalised_bad_plans
    assert len(handler.results) == 0


# Plan Handler ###
@pytest.fixture
def agent_plan_handler(test_bad_plans_config, input_manager):
    handler = plan_handlers.PlanLogs(test_bad_plans_config, 'poor')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)
    assert len(handler.plans_log.chunk) == 0
    return handler


def test_agent_plans_handler(agent_plan_handler):
    handler = agent_plan_handler

    plans = handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert len(handler.plans_log.chunk) == 8


@pytest.fixture
def agent_plans_handler_finalised(agent_plan_handler):
    handler = agent_plan_handler
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    handler.finalise()
    return handler


def test_finalised_plans(agent_plans_handler_finalised):
    handler = agent_plans_handler_finalised

    assert len(handler.results) == 0


# Plans Wrapping case

# Bad Config (plans wrap past 24hrs)
@pytest.fixture
def agent_plans_handler_bad_plans(test_bad_plans_config, input_bad_plans_manager):
    handler = plan_handlers.PlanLogs(test_bad_plans_config, 'poor')

    resources = input_bad_plans_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.plans_log.chunk) == 0
    return handler


# @pytest.fixture
# def agent_plans_handler_finalised_bad_plans(agent_plans_handler_bad_plans):
#     handler = agent_plans_handler_bad_plans
#     plans = handler.resources['plans']
#     for plan in plans.persons:
#         handler.process_plans(plan)
#     assert handler.plans_log.chunk[-1].get('end_day') == 1
#     handler.finalise()
#     return handler


# def test_finalised_plans_bad_plans(agent_plans_handler_finalised_bad_plans):
#     handler = agent_plans_handler_finalised_bad_plans
#     assert len(handler.results) == 0


### Agent Highway Distance Handler ###
@pytest.fixture
def agent_distances_handler_car_mode(test_config, input_manager):
    handler = plan_handlers.AgentHighwayDistanceLogs(test_config, 'car')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.agent_ids) == len(handler.resources['subpopulations'].map)
    assert list(handler.agent_indices.keys()) == handler.agent_ids

    assert len(handler.ways) == len(handler.resources['osm_ways'].classes)
    assert list(handler.ways_indices.keys()) == handler.ways

    assert handler.distances.shape == (
        len(handler.resources['subpopulations'].map),
        len(handler.resources['osm_ways'].classes)
    )

    return handler


def test_agent_distances_handler_car_mode(agent_distances_handler_car_mode):
    handler = agent_distances_handler_car_mode

    plans = handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert np.sum(handler.distances) == 40600.0

    # agent
    assert np.sum(handler.distances[handler.agent_indices['chris']]) == 20400.0

    # class
    assert np.sum(handler.distances[:, handler.ways_indices['trunk']]) == 30600.0


@pytest.fixture
def agent_distances_handler_finalised_car(agent_distances_handler_car_mode):
    handler = agent_distances_handler_car_mode
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    handler.finalise()
    return handler


def test_finalised_agent_distances_car(agent_distances_handler_finalised_car):
    handler = agent_distances_handler_finalised_car

    for name, result in handler.results.items():
        cols = handler.ways
        if 'total' in name:
            for c in cols:
                assert c in result.index
            assert 'total' in result.index
            assert np.sum(result[cols].values) == 40600.0

        else:
            for c in cols:
                assert c in result.columns
            assert 'total' in result.columns
            df = result.loc[:, cols]
            assert np.sum(df.values) == 40600.0


### Trips Highway Distance Handler ###
@pytest.fixture
def trip_distances_handler_car_mode(test_config, input_manager):
    handler = plan_handlers.TripHighwayDistanceLogs(test_config, 'car')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.ways) == len(handler.resources['osm_ways'].classes)

    return handler


def test_trip_distances_handler_car_mode(trip_distances_handler_car_mode):
    handler = trip_distances_handler_car_mode

    plans = handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert len(handler.distances_log.chunk) == 10
    assert sum([d['None'] for d in handler.distances_log.chunk]) == 10000
    assert sum([d['trunk'] for d in handler.distances_log.chunk]) == 30600

    # agent
    assert sum([d['trunk'] for d in handler.distances_log.chunk if d['agent'] == 'chris']) == 20400.0


def test_trip_distances_handler_finalised_car(trip_distances_handler_car_mode):
    handler = trip_distances_handler_car_mode
    plans = handler.resources['plans']
    for plan in plans.persons:
        handler.process_plans(plan)
    handler.finalise()

    path = handler.distances_log.path
    results = pd.read_csv(path)
    assert len(results) == 10
    assert sum(results.trunk) == 30600
    assert sum(results.loc[results.agent == 'chris'].trunk) == 20400


### Modeshare Handler ###
@pytest.fixture
def test_plan_modeshare_handler(test_config, input_manager):
    handler = plan_handlers.ModeShares(test_config, 'all')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    periods = 24

    # assert len(handler.modes) == len(handler.resources['output_config'].modes)
    assert list(handler.mode_indices.keys()) == handler.modes

    assert len(handler.classes) == len(handler.resources['subpopulations'].classes)
    assert list(handler.class_indices.keys()) == handler.classes

    assert len(handler.activities) == len(handler.resources['output_config'].activities)
    assert list(handler.activity_indices.keys()) == handler.activities

    assert handler.mode_counts.shape == (6, 3, 2, 24)

    return handler


def test_distance_mode_share_simple(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler
    string = """
    <person id="nick">
        <plan score="1" selected="yes">
            <activity type="home" link="1-2" x="0.0" y="0.0" end_time="08:00:00" >
            </activity>
            <leg mode="car" dep_time="08:00:00" trav_time="00:00:04">
            <route type="links" start_link="1-2" end_link="1-5" trav_time="00:00:04" distance="10100.0">1-2 2-1 1-5</route>
            </leg>
            <activity type="work" link="1-5" x="0.0" y="10000.0" end_time="17:30:00" >
            </activity>
        </plan>
    </person>
    """
    person = etree.fromstring(string)
    handler.process_plans(person)
    assert np.sum(handler.mode_counts[handler.mode_indices['car']]) == 1


def test_distance_mode_share_pt(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler
    string = """
    <person id="nick">
        <plan score="1" selected="yes">
            <activity type="home" link="1-2" x="0.0" y="0.0" end_time="08:00:00" >
            </activity>
            <leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="1-2" end_link="1-2" trav_time="00:01:18" distance="65.0"></route>
			</leg>
			<activity type="pt interaction" link="1-2" x="50.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="pt" trav_time="00:43:42">
				<route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" distance="10100.0">PT1===home_stop_out===city_line===work_bound===work_stop_in</route>
			</leg>
			<activity type="pt interaction" link="3-4" x="10050.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="3-4" end_link="3-4" trav_time="00:01:18" distance="65.0"></route>
			</leg>
            <activity type="work" link="1-5" x="0.0" y="10000.0" end_time="17:30:00" >
            </activity>
        </plan>
    </person>
    """
    person = etree.fromstring(string)
    handler.process_plans(person)
    assert np.sum(handler.mode_counts[handler.mode_indices['bus']]) == 1


def test_distance_mode_share_complex_pt_1(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler
    handler.resources['transit_schedule'].route_to_mode_map["rail_dummy"] = "rail"
    modes = handler.modes + ['rail']
    handler.modes, handler.mode_indices = handler.generate_id_map(modes)
    handler.mode_counts = np.zeros((len(handler.modes),
                                    len(handler.classes),
                                    len(handler.activities),
                                    handler.config.time_periods))
    string = """
    <person id="nick">
        <plan score="1" selected="yes">
            <activity type="home" link="1-2" x="0.0" y="0.0" end_time="08:00:00" >
            </activity>
            <leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="1-2" end_link="1-2" trav_time="00:01:18" distance="65.0"></route>
			</leg>
			<activity type="pt interaction" link="1-2" x="50.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="pt" trav_time="00:43:42">
				<route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" distance="10100.0">PT1===home_stop_out===city_line===work_bound===work_stop_in</route>
			</leg>
			<activity type="pt interaction" link="3-4" x="10050.0" y="0.0" max_dur="00:00:00" >
			</activity>
            <leg mode="pt" trav_time="00:43:42">
				<route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" distance="10101.0">PT1===home_stop_out===rail_dummy===rail_dummy===work_stop_in</route>
			</leg>
			<activity type="pt interaction" link="3-4" x="10050.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="3-4" end_link="3-4" trav_time="00:01:18" distance="65.0"></route>
			</leg>
            <activity type="work" link="1-5" x="0.0" y="10000.0" end_time="17:30:00" >
            </activity>
        </plan>
    </person>
    """
    person = etree.fromstring(string)
    handler.process_plans(person)
    assert np.sum(handler.mode_counts[handler.mode_indices['rail']]) == 1


def test_distance_mode_share_complex_pt_2(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler
    handler.resources['transit_schedule'].route_to_mode_map["rail_dummy"] = "rail"
    modes = handler.modes + ['rail']
    handler.modes, handler.mode_indices = handler.generate_id_map(modes)
    handler.mode_counts = np.zeros((len(handler.modes),
                                    len(handler.classes),
                                    len(handler.activities),
                                    handler.config.time_periods))
    string = """
    <person id="nick">
        <plan score="1" selected="yes">
            <activity type="home" link="1-2" x="0.0" y="0.0" end_time="08:00:00" >
            </activity>
            <leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="1-2" end_link="1-2" trav_time="00:01:18" distance="65.0"></route>
			</leg>
			<activity type="pt interaction" link="1-2" x="50.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="pt" trav_time="00:43:42">
				<route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" distance="10101.0">PT1===home_stop_out===city_line===work_bound===work_stop_in</route>
			</leg>
			<activity type="pt interaction" link="3-4" x="10050.0" y="0.0" max_dur="00:00:00" >
			</activity>
            <leg mode="pt" trav_time="00:43:42">
				<route type="experimentalPt1" start_link="1-2" end_link="3-4" trav_time="00:43:42" distance="10100.0">PT1===home_stop_out===rail_dummy===rail_dummy===work_stop_in</route>
			</leg>
			<activity type="pt interaction" link="3-4" x="10050.0" y="0.0" max_dur="00:00:00" >
			</activity>
			<leg mode="transit_walk" trav_time="00:01:18">
				<route type="generic" start_link="3-4" end_link="3-4" trav_time="00:01:18" distance="65.0"></route>
			</leg>
            <activity type="work" link="1-5" x="0.0" y="10000.0" end_time="17:30:00" >
            </activity>
        </plan>
    </person>
    """
    person = etree.fromstring(string)
    handler.process_plans(person)
    assert np.sum(handler.mode_counts[handler.mode_indices['bus']]) == 1


def test_plan_handler_test_data(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler

    plans = test_plan_modeshare_handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)

    assert np.sum(handler.mode_counts) == 10

    # mode
    assert np.sum(handler.mode_counts[handler.mode_indices['car']]) == 4
    assert np.sum(handler.mode_counts[handler.mode_indices['bus']]) == 4
    assert np.sum(handler.mode_counts[handler.mode_indices['bike']]) == 2
    assert np.sum(handler.mode_counts[handler.mode_indices['walk']]) == 0

    # class
    assert np.sum(handler.mode_counts[:, handler.class_indices['rich'], :, :]) == 2
    assert np.sum(handler.mode_counts[:, handler.class_indices['poor'], :, :]) == 8
    # assert np.sum(handler.mode_counts[:, handler.class_indices['not_applicable'], :, :]) == 0

    # activities
    # assert np.sum(handler.mode_counts[:, :, handler.activity_indices['pt interaction'], :]) == 0
    assert np.sum(handler.mode_counts[:, :, handler.activity_indices['work']]) == 5
    assert np.sum(handler.mode_counts[:, :, handler.activity_indices['home']]) == 5

    # time
    assert np.sum(handler.mode_counts[:, :, :, :12]) == 5
    assert np.sum(handler.mode_counts[:, :, :, 12:]) == 5


@pytest.fixture
def test_plan_handler_finalised(test_plan_modeshare_handler):
    handler = test_plan_modeshare_handler
    plans = test_plan_modeshare_handler.resources['plans']
    for person in plans.persons:
        handler.process_plans(person)
    handler.finalise()
    return handler


def test_finalised_mode_counts(test_plan_handler_finalised):
    handler = test_plan_handler_finalised

    for name, result in handler.results.items():
        if 'count' in name:
            cols = handler.modes
            if isinstance(result, pd.DataFrame):
                for c in cols:
                    assert c in result.columns
                df = result.loc[:, cols]
                assert np.sum(df.values) == 10 / handler.config.scale_factor

                if 'class' in result.columns:
                    assert set(result.loc[:, 'class']) == set(handler.classes)
                if 'activity' in result.columns:
                    assert set(result.loc[:, 'activity']) == set(handler.activities)
                if 'hour' in result.columns:
                    assert set(result.loc[:, 'hour']) == set(range(24))
            else:
                for c in cols:
                    assert c in result.index
                df = result.loc[cols]
                assert np.sum(df.values) == 10 / handler.config.scale_factor


def test_finalised_mode_shares(test_plan_handler_finalised):
    handler = test_plan_handler_finalised

    for name, result in handler.results.items():
        if 'counts' not in name:
            cols = handler.modes
            if isinstance(result, pd.DataFrame):
                for c in cols:
                    assert c in result.columns
                df = result.loc[:, cols]
                assert np.sum(df.values) == 1

                if 'class' in result.columns:
                    assert set(result.loc[:, 'class']) == set(handler.classes)
                if 'activity' in result.columns:
                    assert set(result.loc[:, 'activity']) == set(handler.activities)
                if 'hour' in result.columns:
                    assert set(result.loc[:, 'hour']) == set(range(24))
            else:
                for c in cols:
                    assert c in result.index
                df = result.loc[cols]
                assert np.sum(df.values) == 1


# Plan Handler Manager
def test_load_plan_handler_manager(test_config, test_paths):
    input_workstation = InputsWorkStation(test_config)
    input_workstation.connect(managers=None, suppliers=[test_paths])
    input_workstation.load_all_tools()
    input_workstation.build()

    plan_workstation = PlanHandlerWorkStation(test_config)
    plan_workstation.connect(managers=None, suppliers=[input_workstation])

    tool = plan_workstation.tools['mode_shares']
    plan_workstation.resources['mode_shares'] = tool(test_config, 'all')

    plan_workstation.build(write_path=test_outputs)

    for handler in plan_workstation.resources.values():
        for name, result in handler.results.items():
            if 'count' not in name:
                cols = handler.modes
                if isinstance(result, pd.DataFrame):
                    for c in cols:
                        assert c in result.columns
                    df = result.loc[:, cols]
                    assert np.sum(df.values) == 1

                    if 'class' in result.columns:
                        assert set(result.loc[:, 'class']) == set(handler.classes)
                    if 'activity' in result.columns:
                        assert set(result.loc[:, 'activity']) == set(handler.activities)
                    if 'hour' in result.columns:
                        assert set(result.loc[:, 'hour']) == set(range(24))
                else:
                    for c in cols:
                        assert c in result.index
                    df = result.loc[cols]
                    assert np.sum(df.values) == 1
