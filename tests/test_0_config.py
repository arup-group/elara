import pytest

from elara.config import Config


def test_config_override_update_input_fields_and_output_path():
    config = Config("tests/test_xml_scenario.toml")
    events = 'output_events.xml.gz'
    network = 'output_network.xml.gz'
    road_pricing = './tests/test_fixtures/road_pricing.xml'
    override = "/test/path"

    config.override(override)

    assert config.settings['inputs']['events'] == override + "/" + events
    assert config.settings['inputs']['network'] == override + "/" + network
    assert config.settings['inputs']['road_pricing'] == road_pricing

    assert config.settings['outputs']['path'] == override + "/" + config.settings['outputs']['path'].split("/")[-1]
    assert config.output_path == override + "/" + config.settings['outputs']['path'].split("/")[-1]


def test_version_11_attributes_path_is_attributes():
    config = Config("tests/test_xml_scenario.toml")

    assert config.attributes_path == "./tests/test_fixtures/output_personAttributes.xml.gz"
    

def test_version_12_attributes_path_is_plans():
    config = Config("tests/test_xml_scenario_v12.toml")

    assert config.attributes_path == "./tests/test_fixtures/output_plans_v12.xml"


def test_roadpricing_config_path():
    config = Config("tests/test_xml_scenario.toml")
    assert config.road_pricing_path == "./tests/test_fixtures/road_pricing.xml"


def test_roadpricing_config_path_missing():
    config = Config("tests/test_xml_scenario_bad_path.toml")
    with pytest.raises(KeyError):
        assert config.road_pricing_path == "./tests/test_fixtures/road_pricing.xml"
