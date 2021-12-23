import sys
import os
from numpy.matrixlib import defmatrix
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree
from datetime import datetime, timedelta


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara import input_plan_handlers, plan_handlers
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
    config_path = os.path.join(test_dir, 'test_xml_scenario_with_input_plans.toml')
    config = Config(config_path)
    assert config
    return config


@pytest.fixture
def test_config_v12():
    config_path = os.path.join(test_dir, 'test_xml_scenario_v12_with_input_plans.toml')
    config = Config(config_path)
    assert config
    return config

@pytest.fixture
def test_config_v13():
    config_path = os.path.join(test_dir, 'test_xml_scenario_v13_with_input_plans.toml')
    config = Config(config_path)
    assert config
    return config

def test_v12_config(test_config_v12):
    assert test_config_v12.version == 12

@pytest.fixture
def test_paths_v12(test_config_v12):
    paths = PathFinderWorkStation(test_config_v12)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths

@pytest.fixture
def input_manager_v12(test_config_v12, test_paths_v12):
    input_workstation = InputsWorkStation(test_config_v12)
    input_workstation.connect(managers=None, suppliers=[test_paths_v12])
    input_workstation.load_all_tools()
    input_workstation.build()
    return input_workstation

# Normal Case
@pytest.fixture
def agent_trip_handler(test_config_v12, input_manager_v12):
    input_manager = input_manager_v12
    handler = input_plan_handlers.InputTripLogs(test_config_v12, 'all')

    resources = input_manager.resources
    handler.build(resources, write_path=test_outputs)

    assert len(handler.activities_log.chunk) == 0
    assert len(handler.trips_log.chunk) == 0

    return handler

def test_agent_input_trip_log_process_person(agent_trip_handler):
    handler = agent_trip_handler

    person = """
    <person id="nick">
        <plan>
            <activity type="home" link="1-2" x="0.0" y="0.0" end_time="08:00:00" ></activity>
            <leg mode="car" dep_time="08:00:00" trav_time="00:00:04"></leg>
            <activity type="work" link="1-5" x="0.0" y="10000.0" end_time="17:30:00" ></activity>
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