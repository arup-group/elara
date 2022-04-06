import os
import pytest
import pandas as pd
import pyarrow as pa
import lxml.etree as etree

from elara.config import Config, PathFinderWorkStation
from elara import inputs
from elara import event_handlers
from elara.event_handlers import EventHandlerWorkStation, VehicleLinksAnimate

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


# Config
@pytest.fixture
def config():
    config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
    config = Config(config_path)
    return config


# Paths
@pytest.fixture
def paths_workstation(config):
    paths = PathFinderWorkStation(config)
    paths.connect(managers=None, suppliers=None)
    paths.load_all_tools()
    paths.build()
    assert set(paths.resources) == set(paths.tools)
    return paths


# Input Manager
@pytest.fixture
def inputs_workstation(config, paths_workstation):
    input_workstation = inputs.InputsWorkStation(config)
    input_workstation.connect(managers=None, suppliers=[paths_workstation])
    input_workstation.load_all_tools()
    input_workstation.build()
    return input_workstation


@pytest.fixture
def vehicle_links_animate_handler(config, inputs_workstation):
    handler = VehicleLinksAnimate(config=config, mode="all")
    resources = inputs_workstation.resources
    handler.build(resources, write_path=test_outputs)
    return handler


@pytest.fixture
def simple_car_events():
    string = """
        <events>
        <event time = "0.0" type = "vehicle enters traffic" person = "chris"
        link = "1-2" vehicle = "chris" networkMode = "car" relativePosition = "1.0" />
        <event time="1.0" type="left link" vehicle="chris" link="1-2"  />
        <event time="1.0" type="entered link" vehicle="chris" link="2-3"  />
        <event time="2.0" type="left link" vehicle="chris" link="2-3"  />
        </events>
        """
    return etree.fromstring(string)


@pytest.fixture
def complete_car_events():
    string = """
        <events>
        <event time = "0.0" type = "vehicle enters traffic" person = "chris"
        link = "1-2" vehicle = "chris" networkMode = "car" relativePosition = "1.0" />
        <event time="1.0" type="left link" vehicle="chris" link="1-2"  />
        <event time="1.0" type="entered link" vehicle="chris" link="2-3"  />
        <event time="2.0" type="left link" vehicle="chris" link="2-3"  />
        <event time="2.0" type="entered link" vehicle="chris" link="3-4"  />
        <event time="3.0" type="vehicle leaves traffic" person="chris"
        link="3-4" vehicle="chris" networkMode="car" relativePosition="1.0"  />
        </events>
        """
    return etree.fromstring(string)


test_colors = [
    ("bus", [255,40,40]),
    ("Sdvasfdvafafdvvc", [128,128,128]),
]
@pytest.mark.parametrize("mode,color", test_colors)
def test_get_color(mode, color):
    handler = VehicleLinksAnimate(config=Config(), mode="all")
    assert handler.get_color(mode) == color


def test_get_coords(vehicle_links_animate_handler):
    handler = vehicle_links_animate_handler

    assert handler.get_entry_coords("1-2") == [49.766807, -7.55716]
    assert handler.get_exit_coords("1-2") == [49.766874, -7.555778]


def test_vehicle_links_animate_handler_simple_case(
    vehicle_links_animate_handler,
    simple_car_events
    ):
    handler = vehicle_links_animate_handler
    for elem in simple_car_events:
        handler.process_event(elem)
    assert handler.vehicles == {"chris": {"veh_mode": "car", "color": [200,200,200]}}
    assert handler.traces == {"chris": [
        [49.766807, -7.55716], [49.766874, -7.555778], [49.773375, -7.418928]
    ]}
    assert handler.timestamps == {"chris": [1649116800, 1649116801, 1649116802]}
    assert len(handler.vehicle_trips) == 0


def test_vehicle_links_animate_handler_complete_case(
    vehicle_links_animate_handler,
    complete_car_events
    ):
    handler = vehicle_links_animate_handler
    for elem in complete_car_events:
        handler.process_event(elem)

    assert handler.vehicles == {}
    assert handler.traces == {}
    assert handler.timestamps == {}

    assert len(handler.vehicle_trips) == 1
    assert handler.vehicle_trips.chunk == [{
            "veh_mode": "car",
            "color": [200,200,200],
            "path": [
                [49.766807, -7.55716],
                [49.766874, -7.555778],
                [49.773375, -7.418928],
                [49.77344, -7.417546]
                ],
            "timestamps": [1649116800, 1649116801, 1649116802, 1649116803],
            "vid": "chris",
            }]


def test_vehicle_links_animate_handler_finalise(
    vehicle_links_animate_handler,
    complete_car_events
    ):
    handler = vehicle_links_animate_handler
    for elem in complete_car_events:
        handler.process_event(elem)

    handler.finalise()
    reader = pa.ipc.RecordBatchStreamReader(
        os.path.join(test_outputs, 'vehicle_links_animate_all.arrow')
        )
    result = reader.read_all().to_pandas()
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
            "veh_mode": ["car"],
            "color": [[200,200,200]],
            "path": [[
                [49.766807, -7.55716],
                [49.766874, -7.555778],
                [49.773375, -7.418928],
                [49.77344, -7.417546]
                ]],
            "timestamps": [[1649116800, 1649116801, 1649116802, 1649116803]],
            "vid": ["chris"],
            }
        )
    )