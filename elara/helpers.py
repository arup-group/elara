from pathlib import Path
import click


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
