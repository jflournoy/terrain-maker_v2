"""core re-exports functions from other modules; they must be the same objects."""

import importlib

import pytest

import terrain_maker.terrain.core as core


@pytest.mark.parametrize("name", sorted(core._REEXPORTS))
def test_reexport_is_the_home_function(name):
    home = importlib.import_module(core._REEXPORTS[name])
    assert getattr(core, name) is getattr(home, name)

