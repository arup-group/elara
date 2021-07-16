import sys
import os
import pytest
import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon
import logging


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, RequirementsWorkStation, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.plan_handlers import PlanHandlerWorkStation
from elara.event_handlers import EventHandlerWorkStation
from elara.postprocessing import PostProcessWorkStation
from elara.benchmarking import BenchmarkWorkStation
from elara import factory

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
def test_config():
    config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
    config = Config(config_path)
    assert config
    return config


# Config
@pytest.fixture
def test_config_dependencies():
    config_path = os.path.join(test_dir, 'test_xml_scenario_dependencies.toml')
    config = Config(config_path)
    assert config
    return config


def test_requirements_workstation(test_config):
    requirements = RequirementsWorkStation(test_config)
    assert requirements.gather_manager_requirements() == {
        'mode_shares': {'modes':['all']},
        'link_passenger_counts': {'modes':['bus', 'train']},
        'stop_passenger_counts': {'modes':['bus', 'train']},
        'vkt': {'modes':['car']},
        'link_vehicle_counts': {'modes':['car']},
        'test_link_cordon': {'modes':['car']},
    }


def test_requirements(test_config):
    requirements = RequirementsWorkStation(test_config)
    postprocessing = PostProcessWorkStation(test_config)
    benchmarks = BenchmarkWorkStation(test_config)
    event_handlers = EventHandlerWorkStation(test_config)
    plan_handlers = PlanHandlerWorkStation(test_config)
    input_workstation = InputsWorkStation(test_config)
    paths = PathFinderWorkStation(test_config)

    requirements.connect(
        managers=None,
        suppliers=[postprocessing, benchmarks, event_handlers, plan_handlers]
    )
    benchmarks.connect(
        managers=[requirements],
        suppliers=[event_handlers, plan_handlers],
    )
    postprocessing.connect(
        managers=[requirements],
        suppliers=[event_handlers, plan_handlers]
    )
    event_handlers.connect(
        managers=[postprocessing, benchmarks, requirements],
        suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[requirements, benchmarks, postprocessing],
        suppliers=[input_workstation]
    )
    input_workstation.connect(
        managers=[event_handlers, plan_handlers],
        suppliers=[paths]
    )
    paths.connect(
        managers=[input_workstation],
        suppliers=None
    )


# Setup
@pytest.fixture
def requirements(test_config):
    requirements = RequirementsWorkStation(test_config)
    postprocessing = PostProcessWorkStation(test_config)
    benchmarks = BenchmarkWorkStation(test_config)
    event_handlers = EventHandlerWorkStation(test_config)
    plan_handlers = PlanHandlerWorkStation(test_config)
    input_workstation = InputsWorkStation(test_config)
    paths = PathFinderWorkStation(test_config)

    requirements.connect(
        managers=None,
        suppliers=[postprocessing, benchmarks, event_handlers, plan_handlers]
    )
    benchmarks.connect(
        managers=[requirements],
        suppliers=[event_handlers, plan_handlers],
    )
    postprocessing.connect(
        managers=[requirements],
        suppliers=[event_handlers, plan_handlers]
    )
    event_handlers.connect(
        managers=[postprocessing, benchmarks, requirements],
        suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[requirements, benchmarks, postprocessing],
        suppliers=[input_workstation]
    )
    input_workstation.connect(
        managers=[event_handlers, plan_handlers],
        suppliers=[paths]
    )
    paths.connect(
        managers=[input_workstation],
        suppliers=None
    )
    return requirements


def test_dfs(requirements):
    factory.build_graph_depth(requirements)
    assert requirements.depth == 0

    assert requirements.suppliers[0].depth == 1
    assert requirements.suppliers[1].depth == 1
    assert requirements.suppliers[2].depth == 2
    assert requirements.suppliers[3].depth == 2

    assert requirements.suppliers[0].suppliers[0].depth == 2
    assert requirements.suppliers[0].suppliers[1].depth == 2

    assert requirements.suppliers[0].suppliers[0].suppliers[0].depth == 3
    assert requirements.suppliers[0].suppliers[0].suppliers[0].suppliers[0].depth == 4


def test_bfs(requirements):
    factory.build(requirements, write_path=test_outputs)
    assert requirements.resources == {}
    assert set(requirements.suppliers[0].resources) == set({'vkt:car': factory.Tool})
    assert set(requirements.suppliers[2].resources) == set(
        {
            'stop_passenger_counts:train': factory.Tool,
            'stop_passenger_counts:bus': factory.Tool,
            'link_passenger_counts:train': factory.Tool,
            'link_passenger_counts:bus': factory.Tool,
            'link_vehicle_counts:car': factory.Tool,
        }
    )


#test_config_dependencies
# Setup
@pytest.fixture
def requirements_depends(test_config_dependencies):
    requirements = RequirementsWorkStation(test_config_dependencies)
    postprocessing = PostProcessWorkStation(test_config_dependencies)
    benchmarks = BenchmarkWorkStation(test_config_dependencies)
    event_handlers = EventHandlerWorkStation(test_config_dependencies)
    plan_handlers = PlanHandlerWorkStation(test_config_dependencies)
    input_workstation = InputsWorkStation(test_config_dependencies)
    paths = PathFinderWorkStation(test_config_dependencies)

    requirements.connect(
        managers=None,
        suppliers=[postprocessing, benchmarks, event_handlers, plan_handlers]
    )
    benchmarks.connect(
        managers=[requirements],
        suppliers=[event_handlers, plan_handlers],
    )
    postprocessing.connect(
        managers=[requirements],
        suppliers=[event_handlers, plan_handlers]
    )
    event_handlers.connect(
        managers=[postprocessing, benchmarks, requirements],
        suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[requirements, benchmarks, postprocessing],
        suppliers=[input_workstation]
    )
    input_workstation.connect(
        managers=[event_handlers, plan_handlers],
        suppliers=[paths]
    )
    paths.connect(
        managers=[input_workstation],
        suppliers=None
    )
    return requirements


def test_bfs_depends(requirements_depends):
    requirements = requirements_depends
    factory.build(requirements, write_path=test_outputs)
    assert requirements.resources == {}
    assert set(requirements.suppliers[0].resources) == set(
        {
            'vkt:car': factory.Tool,
            'vkt:bus': factory.Tool,
        }
    )
    assert set(requirements.suppliers[2].resources) == set(
        {
            'stop_passenger_counts:train': factory.Tool,
            'stop_passenger_counts:bus': factory.Tool,
            'link_passenger_counts:train': factory.Tool,
            'link_passenger_counts:bus': factory.Tool,
            'link_vehicle_counts:car': factory.Tool,
            'link_vehicle_counts:bus': factory.Tool,
        }
    )


def test_cycle_simple():
    class Node:
        def __init__(self):
            self.suppliers = []

        def connect(self, suppliers):
            self.suppliers = suppliers

    a = Node()
    b = Node()
    a.connect([b])
    b.connect([a])

    assert factory.is_cyclic(a)


def test_cycle_2():
    class Node:
        def __init__(self):
            self.suppliers = []

        def connect(self, suppliers):
            self.suppliers = suppliers

    a = Node()
    b = Node()
    c = Node()
    a.connect([b])
    b.connect([c])
    c.connect([b])

    assert factory.is_cyclic(a)


def test_cycle_3():
    class Node:
        def __init__(self):
            self.suppliers = []

        def connect(self, suppliers):
            self.suppliers = suppliers

    a = Node()
    b = Node()
    c = Node()
    d = Node()
    a.connect([b])
    b.connect([c])
    c.connect([d])
    d.connect([b])

    assert factory.is_cyclic(a)


def test_not_cycle_simple():
    class Node:
        def __init__(self):
            self.suppliers = []

        def connect(self, suppliers):
            self.suppliers = suppliers

    a = Node()
    b = Node()
    c = Node()
    a.connect([b])
    b.connect([c])

    assert not factory.is_cyclic(a)


def test_not_cycle_2():
    class Node:
        def __init__(self):
            self.suppliers = []

        def connect(self, suppliers):
            self.suppliers = suppliers

    a = Node()
    b = Node()
    c = Node()
    a.connect([b, c])
    b.connect([c])

    assert not factory.is_cyclic(a)


def test_not_cycle_3():
    class Node:
        def __init__(self):
            self.suppliers = []

        def connect(self, suppliers):
            self.suppliers = suppliers

    a = Node()
    b = Node()
    c = Node()
    d = Node()
    a.connect([b, c, d])
    b.connect([c, d])

    assert not factory.is_cyclic(a)


def test_broken():
    class Node:
        def __init__(self):
            self.suppliers = []
            self.managers = []

        def connect(self, suppliers, managers):
            self.suppliers = suppliers
            self.managers = managers

    a = Node()
    b = Node()
    c = Node()
    a.connect([b], [a])
    b.connect([c], None)

    assert factory.is_broken(a)


def test_not_broken():
    class Node:
        def __init__(self):
            self.suppliers = []
            self.managers = []

        def connect(self, suppliers, managers):
            self.suppliers = suppliers
            self.managers = managers

    a = Node()
    b = Node()
    c = Node()
    a.connect([b], None)
    b.connect([c], [a])
    c.connect(None, [b])

    assert not factory.is_broken(a)


def test_convert_to_unique_keys():

    keys = factory.convert_to_unique_keys(
        {"req1":[1,2], "req2":[1], "req3":None}
    )
    assert keys == ['req1:1', 'req1:2', 'req2:1', 'req3']


def test_write_geojson(tmpdir):
    df = pd.DataFrame({1:[1,2,3], 2: [4,5,6]})
    poly = Polygon(((0,0), (1,0), (1,1), (0,1)))
    gdf = gp.GeoDataFrame(df, geometry=[poly]*3)
    workstation = factory.WorkStation(config=None)
    workstation.write_geojson(
        write_object=gdf,
        name='test.geojson',
        write_path=tmpdir
    )
    path = os.path.join(tmpdir, 'test.geojson')
    assert os.path.exists(path)


def test_write_geojson_no_path(tmpdir):
    df = pd.DataFrame({1:[1,2,3], 2: [4,5,6]})
    poly = Polygon(((0,0), (1,0), (1,1), (0,1)))
    gdf = gp.GeoDataFrame(df, geometry=[poly]*3)
    class DummyConfig:
        output_path = tmpdir
    config = DummyConfig()
    workstation = factory.WorkStation(config=config)
    workstation.config.output_path
    workstation.write_geojson(
        write_object=gdf,
        name='test.geojson',
        write_path=None
    )
    path = os.path.join(tmpdir, 'test.geojson')
    assert os.path.exists(path)


def test_write_json(tmpdir):
    data = {1:[1,2,3], 2: [4,5,6]}
    workstation = factory.WorkStation(config=None)
    workstation.write_json(
        write_object=data,
        name='test.json',
        write_path=tmpdir
    )
    path = os.path.join(tmpdir, 'test.json')
    assert os.path.exists(path)


def test_write_json_no_path(tmpdir):
    data = {1:[1,2,3], 2: [4,5,6]}
    class DummyConfig:
        output_path = tmpdir
    config = DummyConfig()
    workstation = factory.WorkStation(config=config)
    workstation.write_json(
        write_object=data,
        name='test.json',
        write_path=None
    )
    path = os.path.join(tmpdir, 'test.json')
    assert os.path.exists(path)


def test_write_geojson_tool(tmpdir):
    df = pd.DataFrame({1:[1,2,3], 2: [4,5,6]})
    poly = Polygon(((0,0), (1,0), (1,1), (0,1)))
    gdf = gp.GeoDataFrame(df, geometry=[poly]*3)
    tool = factory.Tool(config=None)
    tool.logger = logging.getLogger(__name__)
    tool.write_geojson(
        write_object=gdf,
        name='test2.geojson',
        write_path=tmpdir
    )
    path = os.path.join(tmpdir, 'test2.geojson')
    assert os.path.exists(path)


def test_write_geojson_tool_no_path(tmpdir):
    df = pd.DataFrame({1:[1,2,3], 2: [4,5,6]})
    poly = Polygon(((0,0), (1,0), (1,1), (0,1)))
    gdf = gp.GeoDataFrame(df, geometry=[poly]*3)
    class DummyConfig:
        output_path = tmpdir
    config = DummyConfig()
    tool = factory.Tool(config=config)
    tool.logger = logging.getLogger(__name__)
    tool.write_geojson(
        write_object=gdf,
        name='test2.geojson',
        write_path=None
    )
    path = os.path.join(tmpdir, 'test2.geojson')
    assert os.path.exists(path)


def test_write_json_tool(tmpdir):
    data = {1:[1,2,3], 2: [4,5,6]}
    tool = factory.Tool(config=None)
    tool.logger = logging.getLogger(__name__)
    tool.write_json(
        write_object=data,
        name='test2.json',
        write_path=tmpdir
    )
    path = os.path.join(tmpdir, 'test2.json')
    assert os.path.exists(path)


def test_write_json_tool_no_path(tmpdir):
    data = {1:[1,2,3], 2: [4,5,6]}
    class DummyConfig:
        output_path = tmpdir
    config = DummyConfig()
    tool = factory.Tool(config=config)
    tool.logger = logging.getLogger(__name__)
    tool.write_json(
        write_object=data,
        name='test2.json',
        write_path=None
    )
    path = os.path.join(tmpdir, 'test2.json')
    assert os.path.exists(path)


def test_write_png_tool(tmpdir):
    df = pd.DataFrame({1:[1,2,3], 2: [4,5,6]})
    ax = df.plot()
    fig = ax.get_figure()   
    tool = factory.Tool(config=None)
    tool.logger = logging.getLogger(__name__)
    tool.write_png(
        write_object=fig,
        name='test2.png',
        write_path=tmpdir
    )
    path = os.path.join(tmpdir, 'test2.png')
    assert os.path.exists(path)


def test_write_png_tool(tmpdir):
    df = pd.DataFrame({1:[1,2,3], 2: [4,5,6]})
    ax = df.plot()
    fig = ax.get_figure()
    class DummyConfig:
        output_path = tmpdir
    config = DummyConfig()
    tool = factory.Tool(config=config)
    tool.logger = logging.getLogger(__name__)
    tool.write_png(
        write_object=fig,
        name='test2.png',
        write_path=None
    )
    path = os.path.join(tmpdir, 'test2.png')
    assert os.path.exists(path)
