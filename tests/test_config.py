from elara.config import Config

def test_config_override_update_input_fields_and_output_path():
    config = Config("tests/test_xml_scenario.toml")
    events = 'output_events.xml.gz'
    network = 'output_network.xml.gz'
    override = "/test/path"

    config.override(override)

    assert config.parsed_toml['inputs']['events'] == override + "/" + events
    assert config.parsed_toml['inputs']['network'] == override + "/" + network

    assert config.parsed_toml['outputs']['path'] == override
    assert config.output_path == override

