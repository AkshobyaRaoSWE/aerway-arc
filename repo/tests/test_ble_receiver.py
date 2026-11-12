import asyncio

import pytest

from arc.ble_receiver import run_ble_capture
from arc.session import Session
from test_acquisition import contract, frame, metadata


class FakeClient:
    packets = []
    disconnect = False

    def __init__(self, address, *, disconnected_callback, services):
        self.callback = disconnected_callback
        self.is_connected = True

    async def __aenter__(self): return self
    async def __aexit__(self, *args): self.is_connected = False

    async def start_notify(self, uuid, callback):
        for packet in self.packets: callback(None, packet)
        if self.disconnect:
            self.is_connected = False
            self.callback(self)

    async def stop_notify(self, uuid): pass


def test_ble_software_transport_records(tmp_path):
    c = contract(True)
    class Client(FakeClient): packets = [frame(c)]
    path = asyncio.run(run_ble_capture(tmp_path / "record", metadata(True), c, "fixture", duration_s=.01, client_factory=Client))
    assert Session(path).complete
    assert Session(path).read("emg", 0, .02)[1].all()


@pytest.mark.parametrize("fault", ["overflow", "disconnect"])
def test_ble_faults_leave_incomplete(tmp_path, fault):
    c = contract(True)
    class Client(FakeClient):
        packets = [frame(c)] * 3
        disconnect = fault == "disconnect"
    with pytest.raises(RuntimeError, match=fault if fault == "overflow" else "disconnected"):
        asyncio.run(run_ble_capture(tmp_path / fault, metadata(True), c, "fixture", duration_s=.01,
                                    queue_size=1 if fault == "overflow" else 4, client_factory=Client))
    assert not Session(tmp_path / fault).complete


def test_unconfirmed_never_connects(tmp_path):
    with pytest.raises(ValueError, match="unconfirmed"):
        asyncio.run(run_ble_capture(tmp_path / "no", metadata(True), contract(), "fixture", duration_s=1))
    assert not (tmp_path / "no").exists()
