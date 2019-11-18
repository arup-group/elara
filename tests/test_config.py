from elara.config import Config

def test_config_override():
    config = Config("test_xml_scenario.toml")
    event = 'events'
    network = 'network'
    path = 'output'
    overrides = "{" \
                    "'events': '{}'," \
                    "'network': '{}'" \
                    "'path': '{}'" \
                "}".format(event, network, path)

    config.override(overrides)

    assert config.parsed_toml['input']['events'] == event
    assert config.parsed_toml['input']['network'] == network

    assert config.parsed_toml['outputs']['path'] == path
    assert config.output_path == path

