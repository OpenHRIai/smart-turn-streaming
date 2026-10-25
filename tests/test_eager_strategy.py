"""TurnAnalyzerEagerUserTurnStopStrategy against a scripted analyzer: the speculation starts on the
analyzer's eager signal, is confirmed by the commit, and is withdrawn on a resumption, a timeout, a
transcript mismatch or an unresolved turn end."""

from __future__ import annotations

import asyncio

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager

from eot.pipecat_eager_strategy import TurnAnalyzerEagerUserTurnStopStrategy


class Scripted(BaseTurnAnalyzer):
    """append_audio pops the next verdict from ``script``: "inc", "eager" (INCOMPLETE + eager) or "complete"."""

    def __init__(self, script):
        super().__init__()
        self.script = list(script)
        self._eager = False
        self.cleared = 0

    @property
    def speech_triggered(self):
        return True

    @property
    def params(self):
        return BaseTurnParams()

    def append_audio(self, buffer, is_speech):
        v = self.script.pop(0) if self.script else "inc"
        if v == "eager":
            self._eager = True
        return EndOfTurnState.COMPLETE if v == "complete" else EndOfTurnState.INCOMPLETE

    def pop_eager(self):
        e, self._eager = self._eager, False
        return e

    async def analyze_end_of_turn(self):
        return EndOfTurnState.INCOMPLETE, None

    def clear(self):
        self.cleared += 1


class Events:
    def __init__(self, s):
        self.inferences, self.cancels, self.stops = [], [], []

        @s.event_handler("on_user_turn_inference_triggered")
        async def _i(strategy, speculation):
            self.inferences.append(speculation)

        @s.event_handler("on_user_turn_speculation_cancelled")
        async def _c(strategy):
            self.cancels.append(True)

        @s.event_handler("on_user_turn_stopped")
        async def _s(strategy, params):
            self.stops.append(params)


async def make(script, **kw):
    an = Scripted(script)
    s = TurnAnalyzerEagerUserTurnStopStrategy(turn_analyzer=an, wait_for_transcript=False, **kw)
    await s.setup(FrameProcessorSetup(clock=None, task_manager=TaskManager(), pipeline_worker=None))
    ev = Events(s)
    await s.handle_user_turn_started()
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    return s, an, ev


def audio():
    return InputAudioRawFrame(audio=b"\0" * 640, sample_rate=16000, num_channels=1)


def run(coro_fn):
    """pytest-asyncio is not installed: run each async test on a fresh loop."""
    def wrapper():
        asyncio.run(coro_fn())
    wrapper.__name__ = coro_fn.__name__
    return wrapper


@run
async def test_eager_then_commit_confirms_the_speculation():
    s, an, ev = await make(["inc", "eager", "inc", "complete"])
    for _ in range(4):
        await s.process_frame(audio())
    # one speculative inference; the commit only finalizes (no second inference for the same turn)
    assert len(ev.inferences) == 1
    assert ev.inferences[0] is not None and ev.inferences[0].text == ""
    assert len(ev.stops) == 1 and ev.stops[0].confirms_speculation
    assert not ev.cancels
    assert not s.speculating


@run
async def test_eager_is_raised_once_and_only_when_not_speaking():
    s, an, ev = await make(["eager", "eager", "inc"])
    for _ in range(3):
        await s.process_frame(audio())
    assert len([i for i in ev.inferences if i is not None]) == 1
    assert s.speculating


@run
async def test_resumed_speech_withdraws_the_speculation():
    s, an, ev = await make(["eager", "inc"])
    await s.process_frame(audio())
    assert s.speculating
    await s.process_frame(VADUserStartedSpeakingFrame())
    assert ev.cancels == [True]
    assert not s.speculating and not ev.stops
    # a later silence can speculate again
    await s.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    an.script = ["eager"]
    await s.process_frame(audio())
    assert len([i for i in ev.inferences if i is not None]) == 2


@run
async def test_timeout_withdraws_the_speculation():
    s, an, ev = await make(["eager"], speculation_timeout=0.05)
    await s.process_frame(audio())
    await asyncio.sleep(0.15)
    assert ev.cancels == [True] and not s.speculating and not ev.stops


@run
async def test_transcript_mismatch_discards_and_reruns():
    s, an, ev = await make(["eager", "complete"])
    await s.process_frame(InterimTranscriptionFrame(text="hello", user_id="u", timestamp="t"))
    await s.process_frame(audio())                        # speculates on the interim "hello"
    assert ev.inferences[-1].text == "hello"
    await s.process_frame(TranscriptionFrame(text="hello there", user_id="u", timestamp="t"))
    await s.process_frame(audio())                        # commit
    assert ev.cancels == [True]
    assert len(ev.stops) == 1 and not ev.stops[0].confirms_speculation
    assert ev.inferences[-1] is None                      # inference re-run on the committed text


@run
async def test_unresolved_turn_end_withdraws():
    s, an, ev = await make(["eager"])
    await s.process_frame(audio())
    await s.handle_user_turn_stopped()                    # the watchdog / an interruption ended the turn
    assert ev.cancels == [True] and not s.speculating
    assert an.cleared == 1


@run
async def test_plain_analyzer_without_pop_eager_behaves_as_before():
    class Plain(Scripted):
        pop_eager = None            # getattr(analyzer, "pop_eager", None) -> None: no eager protocol
    an = Plain(["inc", "complete"])
    s =TurnAnalyzerEagerUserTurnStopStrategy(turn_analyzer=an, wait_for_transcript=False)
    await s.setup(FrameProcessorSetup(clock=None, task_manager=TaskManager(), pipeline_worker=None))
    ev = Events(s)
    await s.handle_user_turn_started()
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await s.process_frame(audio()); await s.process_frame(audio())
    assert ev.inferences == [None] and len(ev.stops) == 1 and not ev.stops[0].confirms_speculation
