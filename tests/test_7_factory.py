import sys
import os
import pytest


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, RequirementsWorkStation, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.plan_handlers import PlanHandlerWorkStation
from elara.event_handlers import EventHandlerWorkStation
from elara.postprocessing import PostProcessWorkStation
from elara.benchmarking import BenchmarkWorkStation
from elara import factory
sys.path.append(os.path.abspath('../tests'))


# Config
@pytest.fixture
def test_config():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    assert config
    return config


def test_requirements(test_config):
    requirments = RequirementsWorkStation(test_config)
    assert requirments.get_requirements() == {
        'mode_share': ['all'],
        'passenger_counts': ['bus', 'train'],
        'stop_interactions': ['bus', 'train'],
        'vkt': ['car'],
        'volume_counts': ['car']
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
    factory.build(requirements)
    assert requirements.resources == {}
    assert set(requirements.suppliers[0].resources) == set({'vkt:car': factory.Tool})


def test_cycle():
    class Node:
        def __init__(self):
            self.suppliers = []

        def connect(self, suppliers):
            self.suppliers = suppliers

    a = Node()
    b = Node()
    a.connect([b])
    b.connect([a])

    assert factory.cyclic(a)


def test_not_cycle():
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

    assert not factory.cyclic(a)


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

    assert factory.broken(a)


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

    assert not factory.broken(a)
