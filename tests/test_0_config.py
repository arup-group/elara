import pytest
import os
import json

from elara.config import Config


def test_config_override_update_input_fields_and_output_path():
    config = Config("tests/test_xml_scenario.toml")
    events = 'output_events.xml.gz'
    network = 'output_network.xml.gz'
    road_pricing = './tests/test_fixtures/road_pricing.xml'
    override = "/test/path"

    config.override(override, dump_log=False)

    assert config.settings['inputs']['events'] == os.path.join(override, events)
    assert config.settings['inputs']['network'] == os.path.join(override, network)
    assert config.settings['inputs']['road_pricing'] == road_pricing

    assert config.settings['outputs']['path'] == os.path.join(override, config.settings['outputs']['path'].split("/")[-1])
    assert config.output_path == os.path.join(override, config.settings['outputs']['path'].split("/")[-1])


def test_dump_overrides_to_json(tmpdir):
    config = Config("tests/test_xml_scenario.toml")
    assert os.path.exists(tmpdir)
    path = os.path.join(tmpdir, "cnfg.json")
    config.dump_settings_to_disk(path)
    assert os.path.exists(path)
    with open(path) as file:
        cnfg = json.load(file)
    assert cnfg["inputs"]["events"] == "./tests/test_fixtures/output_events.xml.gz"


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
