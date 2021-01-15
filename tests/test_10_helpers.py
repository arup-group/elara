import pytest

from elara.helpers import camel_to_snake


test_text_data = [
    ("TestTest", "test_test"),
    ("", ""),
    ("Test", "test"),
]


@pytest.mark.parametrize("textin,textout", test_text_data)
def test_camel_to_snake(textin, textout):
    assert camel_to_snake(textin) == textout
