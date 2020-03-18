from pathlib import Path
import click


class PathPath(click.Path):
    """A Click path argument that returns a pathlib Path, not a string"""
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


class NaturalOrderGroup(click.Group):

    def list_commands(self, ctx):
        return self.commands.keys()
