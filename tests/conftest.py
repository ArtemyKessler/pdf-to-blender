import importlib.util
import os
import shutil

import pytest

from tests.synthetic.make_pdf import make


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    return make(tmp_path_factory.mktemp("syn") / "bottle.pdf")


def has_blender() -> bool:
    return bool(
        os.environ.get("BLENDER") or shutil.which("blender") or importlib.util.find_spec("bpy")
    )
