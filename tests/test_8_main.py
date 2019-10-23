import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "/test_outputs")))
sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, RequirementsWorkStation, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.plan_handlers import PlanHandlerWorkStation
from elara.event_handlers import EventHandlerWorkStation
from elara.postprocessing import PostProcessWorkStation
from elara.benchmarking import BenchmarkWorkStation
from elara import factory
sys.path.append(os.path.abspath('../tests'))

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def test_path(tmpdir_factory):
    path = tmpdir_factory.mktemp("data")
    return path


# Config
@pytest.fixture
def test_config(test_path):
    config_path = os.path.join(test_dir, 'test_xml_scenario.toml')
    config = Config(config_path)
    config.output_path = test_path
    # config.output_path = os.path.join(test_dir, '../test_outputs')
    return config


@pytest.mark.skip(reason=None)
def test_main(test_config):
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

    factory.build(requirements)

    path = test_config.output_path.join('test_town_boardings_bus.csv')
    test_town_boardings_bus = pd.read_csv(path)
    assert test_town_boardings_bus.loc[:, [str(h) for h in range(24)]].sum().sum() == 40000

    path = test_config.output_path.join('benchmarks').join('benchmark_scores.csv')
    benchmark_scores = pd.read_csv(path)
    assert benchmark_scores.score.sum() == 0


@pytest.mark.skip(reason=None)
def test_main_ordering_graph(test_config):
    requirements = RequirementsWorkStation(test_config)
    postprocessing = PostProcessWorkStation(test_config)
    benchmarks = BenchmarkWorkStation(test_config)
    event_handlers = EventHandlerWorkStation(test_config)
    plan_handlers = PlanHandlerWorkStation(test_config)
    input_workstation = InputsWorkStation(test_config)
    paths = PathFinderWorkStation(test_config)

    requirements.connect(
        managers=None,
        suppliers=[event_handlers, plan_handlers, postprocessing, benchmarks]
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
        managers=[postprocessing, requirements, benchmarks],
        suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[benchmarks, requirements, postprocessing],
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

    factory.build(requirements)

    path = test_config.output_path.join('test_town_boardings_bus.csv')
    test_town_boardings_bus = pd.read_csv(path)
    assert test_town_boardings_bus.loc[:, [str(h) for h in range(24)]].sum().sum() == 40000

    path = test_config.output_path.join('benchmarks').join('benchmark_scores.csv')
    benchmark_scores = pd.read_csv(path)
    assert benchmark_scores.score.sum() == 0


# Config
@pytest.fixture
def test_config_missing(test_path):
    config_path = os.path.join('tests/test_xml_scenario_missing.toml')
    config = Config(config_path)
    config.output_path = test_path
    return config


@pytest.mark.skip(reason=None)
def test_main_missing_requirement_still_fulfilled(test_config_missing):
    requirements = RequirementsWorkStation(test_config_missing)
    postprocessing = PostProcessWorkStation(test_config_missing)
    benchmarks = BenchmarkWorkStation(test_config_missing)
    event_handlers = EventHandlerWorkStation(test_config_missing)
    plan_handlers = PlanHandlerWorkStation(test_config_missing)
    input_workstation = InputsWorkStation(test_config_missing)
    paths = PathFinderWorkStation(test_config_missing)

    requirements.connect(
        managers=None,
        suppliers=[event_handlers, plan_handlers, postprocessing, benchmarks]
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
        managers=[postprocessing, requirements, benchmarks],
        suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[benchmarks, requirements, postprocessing],
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

    factory.build(requirements)

    path = test_config_missing.output_path.join('test_town_boardings_bus.csv')
    test_town_boardings_bus = pd.read_csv(path)
    assert test_town_boardings_bus.loc[:, [str(h) for h in range(24)]].sum().sum() == 40000

    path = test_config_missing.output_path.join('benchmarks').join('benchmark_scores.csv')
    benchmark_scores = pd.read_csv(path)
    assert benchmark_scores.score.sum() == 0


# Config
@pytest.fixture
def test_config_bad_path(test_path):
    config_path = os.path.join('tests/test_xml_scenario_bad_path.toml')
    config = Config(config_path)
    config.output_path = test_path
    return config


def test_main_bad_input_path(test_config_bad_path):
    requirements = RequirementsWorkStation(test_config_bad_path)
    postprocessing = PostProcessWorkStation(test_config_bad_path)
    benchmarks = BenchmarkWorkStation(test_config_bad_path)
    event_handlers = EventHandlerWorkStation(test_config_bad_path)
    plan_handlers = PlanHandlerWorkStation(test_config_bad_path)
    input_workstation = InputsWorkStation(test_config_bad_path)
    paths = PathFinderWorkStation(test_config_bad_path)

    requirements.connect(
        managers=None,
        suppliers=[event_handlers, plan_handlers, postprocessing, benchmarks]
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
        managers=[postprocessing, requirements, benchmarks],
        suppliers=[input_workstation]
    )
    plan_handlers.connect(
        managers=[benchmarks, requirements, postprocessing],
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

    with pytest.raises(Exception):
        factory.build(requirements)
