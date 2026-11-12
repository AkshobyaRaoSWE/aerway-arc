"""Bounded laptop BLE subscriber for the explicitly confirmed bench contract."""
from __future__ import annotations

import asyncio
import math

from .acquisition import BenchRecorder, validate_contract


async def run_ble_capture(path, metadata, contract, device_address, *, duration_s=None,
                          stop_event=None, queue_size=128, client_factory=None):
    """Capture received window; firmware end-of-stream acknowledgement is unverified.

    Unexpected disconnect, overflow or corrupt frame leaves a recoverable incomplete
    session. Inject client_factory only for transport software tests.
    """
    validate_contract(contract, require_confirmed=True)
    if metadata.get("synthetic") is not False:
        raise ValueError("BLE hardware capture requires synthetic=false")
    if not isinstance(device_address, str) or not device_address.strip():
        raise ValueError("explicit device address required")
    if type(queue_size) is not int or not 1 <= queue_size <= 4096:
        raise ValueError("queue_size must be1..4096")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("duration must be positive")
    if duration_s is None and stop_event is None:
        raise ValueError("duration or explicit stop_event required")
    if client_factory is None:
        from bleak import BleakClient
        client_factory = BleakClient
    recorder = BenchRecorder(path, metadata, contract)
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=queue_size)
    disconnected = asyncio.Event()
    failure = []
    accepting = True

    def disconnected_callback(_client):
        loop.call_soon_threadsafe(disconnected.set)

    def enqueue(data):
        if accepting and not failure:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                failure.append(RuntimeError("BLE receive queue overflow; capture incomplete"))

    def notification(_sender, data):
        # Bleak notification callbacks run on the event loop. Enqueue immediately:
        # scheduling unbounded call_soon tasks would defeat the bounded queue.
        enqueue(bytes(data))

    try:
        async with client_factory(device_address, disconnected_callback=disconnected_callback,
                                  services=[contract["service_uuid"]]) as client:
            await client.start_notify(contract["notify_uuid"], notification)
            deadline = None if duration_s is None else loop.time() + duration_s
            while True:
                if failure:
                    raise failure[0]
                if disconnected.is_set() or not client.is_connected:
                    raise RuntimeError("BLE disconnected; capture incomplete")
                if (stop_event is not None and stop_event.is_set()) or (deadline is not None and loop.time() >= deadline):
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=.05)
                except asyncio.TimeoutError:
                    continue
                # Keep notifications responsive while durable chunk writes run.
                # Shield then await on cancellation, avoiding file-close/write races.
                work = asyncio.create_task(asyncio.to_thread(recorder.feed, data))
                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError:
                    await work
                    raise
            await client.stop_notify(contract["notify_uuid"])
            accepting = False
            if failure:
                raise failure[0]
            if disconnected.is_set() or not client.is_connected:
                raise RuntimeError("BLE disconnected during stop; capture incomplete")
            while not queue.empty():
                recorder.feed(queue.get_nowait())
            return recorder.stop()
    except BaseException as exc:
        accepting = False
        recorder.abort(exc)
        raise
