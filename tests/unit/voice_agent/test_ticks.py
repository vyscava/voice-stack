"""Ticks: the generated sound, and the start/stop controller around the brain."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from voice_agent.ticks import SAMPLE_RATE, BackgroundTicker, tick_frames, tick_samples


def test_one_period_is_a_short_knock_then_silence():
    s = tick_samples(interval_s=0.5)
    assert len(s) == int(0.5 * SAMPLE_RATE)
    knock_n = int(0.035 * SAMPLE_RATE)
    assert np.abs(s[:knock_n]).max() > 0.5
    assert not s[knock_n:].any()


@pytest.mark.asyncio
async def test_frames_are_20ms_48k_mono_and_loop_the_period():
    gen = tick_frames(interval_s=0.1)
    frames = [await gen.__anext__() for _ in range(15)]  # 300 ms = 3 periods
    assert all(f.sample_rate == 48000 and f.num_channels == 1 and f.samples_per_channel == 960 for f in frames)
    pcm = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])
    period = int(0.1 * SAMPLE_RATE)
    assert np.array_equal(pcm[:period], pcm[period : 2 * period])
    assert np.abs(pcm[:200]).max() > 10000


class FakePlayer:
    def __init__(self):
        self.played = []
        self.handles = []

    def play(self, cfg):
        self.played.append(cfg)
        h = FakeHandle()
        self.handles.append(h)
        return h


class FakeHandle:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def _ticker(player, delay=0.03):
    return BackgroundTicker(player, delay_s=delay, volume=0.25, interval_s=0.45)


@pytest.mark.asyncio
async def test_ticks_start_after_the_delay_at_the_set_volume_and_stop():
    player = FakePlayer()
    t = _ticker(player)
    t.start()
    await asyncio.sleep(0.01)
    assert player.played == []  # not over the ack's own audio
    await asyncio.sleep(0.05)
    assert len(player.played) == 1 and player.played[0].volume == 0.25
    t.stop()
    assert player.handles[0].stopped


@pytest.mark.asyncio
async def test_stop_before_the_delay_means_no_tick_at_all():
    player = FakePlayer()
    t = _ticker(player)
    t.start()
    t.stop()
    await asyncio.sleep(0.06)
    assert player.played == []


@pytest.mark.asyncio
async def test_start_twice_plays_once_and_stop_is_idempotent():
    player = FakePlayer()
    t = _ticker(player)
    t.start()
    t.start()
    await asyncio.sleep(0.05)
    t.start()
    assert len(player.played) == 1
    t.stop()
    t.stop()
    t.start()
    await asyncio.sleep(0.05)
    assert len(player.played) == 2, "a stopped ticker can tick again on the next wait"


@pytest.mark.asyncio
async def test_stop_later_stops_unless_restarted_or_stopped_first():
    player = FakePlayer()
    t = _ticker(player, delay=0.0)
    t.start()
    await asyncio.sleep(0.01)
    t.stop_later(0.03)
    await asyncio.sleep(0.01)
    assert not player.handles[0].stopped
    await asyncio.sleep(0.04)
    assert player.handles[0].stopped, "the deadline stops ticks nobody stopped"

    t.start()
    await asyncio.sleep(0.01)
    t.stop_later(0.03)
    t.start()  # a new wait cancels the old deadline
    await asyncio.sleep(0.05)
    assert not player.handles[1].stopped
    t.stop()


@pytest.mark.asyncio
async def test_a_start_can_set_its_own_delay():
    player = FakePlayer()
    t = _ticker(player, delay=5.0)
    t.start(delay_s=0.2)
    await asyncio.sleep(0.02)
    assert player.played == []
    await asyncio.sleep(0.4)
    assert len(player.played) == 1, "the per-start delay, not the configured 5 s"
    t.stop()
