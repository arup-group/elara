import sys
import os
import pytest
import pandas as pd

sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, RequirementsWorkStation, PathFinderWorkStation
from elara.inputs import InputsWorkStation
from elara.plan_handlers import PlanHandlerWorkStation
from elara.event_handlers import EventHandlerWorkStation
from elara.postprocessing import PostProcessWorkStation
from elara.benchmarking import BenchmarkWorkStation
from elara import factory
sys.path.append(os.path.abspath('../tests'))


@pytest.fixture(scope="session")
def test_path(tmpdir_factory):
    path = tmpdir_factory.mktemp("data")
    return path


# Config
@pytest.fixture
def test_config(test_path):
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    config.output_path = test_path
    return config


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



