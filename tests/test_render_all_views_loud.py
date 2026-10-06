"""render_all_views renders every view it can, then raises if any failed.

It used to log a failed view (or a None result) and return the others, so a caller got a
dict silently missing views.
"""

from pathlib import Path

import pytest

from terrain_maker.terrain import pipeline as pl


def _pipeline(outcomes):
    cls = next(c for c in vars(pl).values() if isinstance(c, type) and hasattr(c, "render_all_views"))
    p = object.__new__(cls)
    p._log = lambda *a, **k: None

    def render_view(view):
        out = outcomes[view]
        if isinstance(out, Exception):
            raise out
        return out

    p.render_view = render_view
    return p


def test_all_views_ok():
    p = _pipeline({"north": Path("n.png"), "above": Path("a.png")})
    assert p.render_all_views(["north", "above"]) == {"north": Path("n.png"), "above": Path("a.png")}


def test_failed_views_raise_after_rendering_the_rest():
    rendered = []
    p = _pipeline({"north": RuntimeError("GPU out of memory"), "south": None, "above": Path("a.png")})
    original = p.render_view
    p.render_view = lambda view: rendered.append(view) or original(view)
    with pytest.raises(RuntimeError) as err:
        p.render_all_views(["north", "south", "above"])
    assert rendered == ["north", "south", "above"]
    assert "north" in str(err.value) and "GPU out of memory" in str(err.value) and "south" in str(err.value)
