from elara.config import Config


def test_config_override_update_input_fields_and_output_path():
    config = Config("tests/test_xml_scenario.toml")
    events = 'output_events.xml.gz'
    network = 'output_network.xml.gz'
    override = "/test/path"

    config.override(override)

    assert config.settings['inputs']['events'] == override + "/" + events
    assert config.settings['inputs']['network'] == override + "/" + network

    assert config.settings['outputs']['path'] == override + "/" + config.settings['outputs']['path'].split("/")[-1]
    assert config.output_path == override + "/" + config.settings['outputs']['path'].split("/")[-1]

