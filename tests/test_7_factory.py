import sys
import os
import pytest


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, Requirements, PathWorkStation
from elara.inputs import InputWorkStation
from elara.handlers.plan_handlers import PlanHandlerStation
from elara.handlers.event_handlers import EventHandlerStation
from elara.postprocessing import PostProcessWorkStation
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
    requirments = Requirements(test_config)
    assert requirments.get_requirements() == {
        'mode_share': ['all'],
        'passenger_counts': ['bus', 'train'],
        'stop_interactions': ['bus', 'train'],
        'vkt': ['car'],
        'volume_counts': ['car']
    }


def test_requirements(test_config):
    requirements = Requirements(test_config)
    postprocessing = PostProcessWorkStation(test_config)
    event_handlers = EventHandlerStation(test_config)
    plan_handlers = PlanHandlerStation(test_config)
    input_workstation = InputWorkStation(test_config)
    paths = PathWorkStation(test_config)

    requirements.connect(
        managers=None, suppliers=[postprocessing, event_handlers, plan_handlers]
    )
    postprocessing.connect(
        managers=[requirements], suppliers=[event_handlers, plan_handlers]
    )
    event_handlers.connect(
        managers=[postprocessing, requirements], suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[requirements, postprocessing], suppliers=[input_workstation]
    )
    input_workstation.connect(
        managers=[event_handlers, plan_handlers], suppliers=[paths]
    )
    paths.connect(
        managers=[input_workstation], suppliers=None
    )


# Setup
@pytest.fixture
def requirements(test_config):
    requirements = Requirements(test_config)
    postprocessing = PostProcessWorkStation(test_config)
    event_handlers = EventHandlerStation(test_config)
    plan_handlers = PlanHandlerStation(test_config)
    input_workstation = InputWorkStation(test_config)
    paths = PathWorkStation(test_config)

    requirements.connect(
        managers=None, suppliers=[postprocessing, event_handlers, plan_handlers]
    )
    postprocessing.connect(
        managers=[requirements], suppliers=[event_handlers, plan_handlers]
    )
    event_handlers.connect(
        managers=[postprocessing, requirements], suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[requirements, postprocessing], suppliers=[input_workstation]
    )
    input_workstation.connect(
        managers=[event_handlers, plan_handlers], suppliers=[paths]
    )
    paths.connect(
        managers=[input_workstation], suppliers=None
    )
    return requirements


def test_dfs(requirements):
    factory.build_graph_depth(requirements)
    assert requirements.depth == 0
    assert requirements.suppliers[0].depth == 1
    assert requirements.suppliers[1].depth == 2
    assert requirements.suppliers[2].depth == 2
    assert requirements.suppliers[0].suppliers[0].depth == 2
    assert requirements.suppliers[0].suppliers[1].depth == 2
    assert requirements.suppliers[0].suppliers[0].suppliers[0].depth == 3
    assert requirements.suppliers[0].suppliers[0].suppliers[0].suppliers[0].depth == 4


def test_bfs(requirements):
    factory.operate_workstation_graph(requirements)
    assert requirements.resources == {}
    assert set(requirements.suppliers[0].resources) == set({'vkt:car': factory.Tool})
