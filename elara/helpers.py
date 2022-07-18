from pathlib import Path
import click
import polyline
from shapely.geometry import LineString
import pandas as pd

class PathPath(click.Path):
    """A Click path argument that returns a pathlib Path, not a string"""
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


class NaturalOrderGroup(click.Group):

    def list_commands(self, ctx):
        return self.commands.keys()


def camel_to_snake(text):
    out = ''
    for i, t in enumerate(text):
        if i == 0:
            out = out + t.lower()
        elif t.isupper():
            out = out + "_" + t.lower()
        else:
            out = out + t
    return out


def decode_polyline_to_shapely_linestring(_polyline):
    """
    :param _polyline: google encoded polyline
    :return: shapely.geometry.LineString
    """
    decoded = polyline.decode(_polyline)
    return LineString(decoded)


def longest_numeric(string: str):
    counter = 0
    for i in string:
        if i.isnumeric():
            counter+=1
        else:
            break
    if counter == 0:
        return []
    return int(string[:counter])


def try_sort_on_numeric_index(df: pd.DataFrame):
    index = [longest_numeric(l) for l in df.index.get_level_values(0)]
    if not len(index) == len(df):
        return None
    df["sorter"] = index
    df.sort_values("sorter", inplace=True)
    df.drop("sorter", inplace=True, axis=1)
