from elara.config import Config

def test_config_override_update_input_fields_and_output_path():
    config = Config("tests/test_xml_scenario.toml")
    events = 'events'
    network = 'network'
    path = 'output'
    overrides = "{" \
                    "'events': 'events'," \
                    "'network': 'network'," \
                    "'path': 'output'" \
                "}"

    config.override(overrides)

    assert config.parsed_toml['inputs']['events'] == events
    assert config.parsed_toml['inputs']['network'] == network

    assert config.parsed_toml['outputs']['path'] == path
    assert config.output_path == path

def test_config_override_ignore_extra_fields():
    config = Config("tests/test_xml_scenario.toml")
    events = 'events'
    network = 'network'
    overrides = "{" \
                    "'events': 'events'," \
                    "'network': 'network'," \
                    "'extra': 'extra'" \
                "}"

    config.override(overrides)

    assert config.parsed_toml['inputs']['events'] == events
    assert config.parsed_toml['inputs']['network'] == network

    assert 'extra' not in config.parsed_toml['inputs']

