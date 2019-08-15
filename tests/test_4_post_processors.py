import sys
import os
import pytest
import pandas as pd
import numpy as np
import lxml.etree as etree


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import inputs, postprocessing
sys.path.append(os.path.abspath('../tests'))


def test_generate_period_headers():
    hours = 24
    headers = postprocessing.generate_period_headers(hours)
    assert isinstance(headers, list)
    assert len(headers) == hours
    assert all(isinstance(elem, str) for elem in headers)


@pytest.fixture
def config():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    return Config(config_path)

paths = {
    'network_path': config.network_path,
    'transit_schedule_path': config.transit_schedule_path,
    'transit_vehicles_path': config.transit_vehicles_path,
}


@pytest.fixture
def network(config):
    return inputs.Network(resources)


@pytest.fixture
def transit_schedule(config):
    return inputs.TransitSchedule(
        config.transit_schedule_path, config.crs
            )


@pytest.fixture
def transit_vehicles(config):
    return inputs.TransitVehicles(
            config.transit_vehicles_path
            )


@pytest.fixture
def vkt_post_processor(config, network, transit_schedule, transit_vehicles):
    return postprocessing.VKT(config, network, transit_schedule, transit_vehicles)


def test_vkt_prerequisites(vkt_post_processor):
    assert vkt_post_processor.check_prerequisites()


def test_vkt_run(vkt_post_processor):
    vkt_post_processor.run()



