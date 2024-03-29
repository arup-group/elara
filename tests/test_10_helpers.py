import pytest
from shapely.geometry import LineString

from elara.helpers import camel_to_snake, decode_polyline_to_shapely_linestring, longest_numeric


test_text_data = [
    ("TestTest", "test_test"),
    ("", ""),
    ("Test", "test"),
]


@pytest.mark.parametrize("textin,textout", test_text_data)
def test_camel_to_snake(textin, textout):
    assert camel_to_snake(textin) == textout


def test_decode_polyline_to_linestring():
    ls = decode_polyline_to_shapely_linestring('emmkhtaBwfemhs`@qxhmKheziWpxhmKieziW')
    assert ls == LineString([(529705.93507, 180572.3558), (529771.19356, 180444.74903), (529705.93507, 180572.3558)])


test_string_data = [
    ("123", 123),
    ("12M", 12),
    ("AAA1", []),
    ("", []),
]
@pytest.mark.parametrize("string,number", test_string_data)
def test_longest_numeric(string, number):
    assert longest_numeric(string) == number