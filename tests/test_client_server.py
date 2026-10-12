"""Client-server mode against a real local pvserver (the Trillium use case).

Checks plugin loading on both sides, server-side file access, camera following from the
client, and that rotated 2D maps keep their orientation when delivered to the client.
"""

import os
import shutil
import signal
import socket
import subprocess
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not (shutil.which("pvserver") and shutil.which("pvpython")), reason="needs pvserver")
def test_client_server(snapdir, tmp_path):
    port = free_port()
    server = subprocess.Popen(["pvserver", f"--server-port={port}", "--force-offscreen-rendering"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        for _ in range(100):  # wait until it listens
            with socket.socket() as s:
                if s.connect_ex(("localhost", port)) == 0:
                    break
            time.sleep(0.1)
        time.sleep(0.5)
        client = subprocess.run(["pvpython", "--force-offscreen-rendering", f"{HERE}/cs_client.py", str(port),
                                 f"{snapdir}/single/snapshot_000.hdf5", str(tmp_path)],
                                capture_output=True, text=True, timeout=240)
        log = client.stdout + client.stderr
        assert "CS_OK" in log, log[-4000:]
    finally:
        os.killpg(server.pid, signal.SIGKILL)


@pytest.mark.skipif(not (shutil.which("pvserver") and shutil.which("mpiexec")), reason="needs MPI pvserver")
def test_mpi_pvserver(snapdir):
    """mpiexec -n 3 pvserver: every rank runs the reader; results must not be duplicated."""
    port = free_port()
    server = subprocess.Popen(["mpiexec", "--oversubscribe", "-n", "3", "pvserver", f"--server-port={port}",
                               "--force-offscreen-rendering"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        for _ in range(200):
            with socket.socket() as s:
                if s.connect_ex(("localhost", port)) == 0:
                    break
            time.sleep(0.1)
        time.sleep(1.0)
        client = subprocess.run(["pvpython", f"{HERE}/mpi_client.py", str(port), f"{snapdir}/single/snapshot_000.hdf5"],
                                capture_output=True, text=True, timeout=240)
        log = client.stdout + client.stderr
        assert "MPI_OK" in log, log[-4000:]
    finally:
        os.killpg(server.pid, signal.SIGKILL)
