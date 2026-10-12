import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root -> pvgizmo
sys.path.insert(0, HERE)  # snapgen

import pvgizmo  # noqa: E402,F401  (sets the multiprocessing executable before meshoid loads)
import snapgen  # noqa: E402


@pytest.fixture(scope="session")
def snapdir(tmp_path_factory):
    return snapgen.make_standard_set(str(tmp_path_factory.mktemp("snaps")))
