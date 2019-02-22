import sys
import os

sys.path.append(os.path.abspath('../elara'))
from elara.config import Config
from elara import inputs
sys.path.append(os.path.abspath('../tests'))


# Config
def test_config_of_xml_inputs():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    assert config


def test_config_of_gzip_inputs():
    config_path = os.path.join('tests/test_gzip_scenario.toml')
    config = Config(config_path)
    assert config


# Events
def test_loading_xml_events():
    config_path = os.path.join('tests/test_xml_scenario.toml')
    config = Config(config_path)
    events = inputs.Events(config.events_path).event_elems
    event = next(events)
    assert event is not None


def test_loading_gzip_events():
    config_path = os.path.join('tests/test_gzip_scenario.toml')
    config = Config(config_path)
    events = inputs.Events(config.events_path).event_elems
    event = next(events)
    assert event is not None


# TODO add other inputs
