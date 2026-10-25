"""A turn-analyzer stop strategy that speculates on an eager end of turn (Stage C1).

Pipecat's speculation path (``UserTurnSpeculation`` -> the aggregator's provisional inference ->
the LLM service's ``SpeculationGate``) is driven today only by STT services that predict an end
of turn before committing to it (``EagerUserTurnStopStrategy``). This strategy gives the same
path to a *turn analyzer*: one that, besides the committed COMPLETE / INCOMPLETE verdict, reports
an **eager** end of turn when its evidence is good enough to start generating but not yet good
enough to end the turn.

    eager   analyzer.pop_eager() is True after append_audio()  -> speculative inference, response held
    commit  append_audio() / analyze_end_of_turn() == COMPLETE -> turn ends; the held response is
                                                                   released (confirms_speculation=True)
    cancel  the VAD reports the user speaking again, the turn ends without a commit, or
            ``speculation_timeout`` elapses                    -> EagerEndOfTurnCancelFrame, response discarded

The analyzer protocol is duck-typed on purpose: any ``BaseTurnAnalyzer`` with a ``pop_eager()``
method (``eot.pipecat_analyzer.StreamingEOTAnalyzer`` has one) can drive it; without the method the
strategy behaves exactly like ``TurnAnalyzerUserTurnStopStrategy``. A proposed upstream shape is a
``BaseTurnAnalyzer.pop_eager()`` returning False by default.

Transcript matching: as in the STT-driven strategy, the committed transcript is compared with the
one the speculation ran on, under ``match_policy`` (default ``NormalizedMatch``); a mismatch
discards the response and runs inference again on the committed text. The speculation text is the
latest transcript this strategy has seen, final or interim, at the moment of the eager signal.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult, UserTurnSpeculation
from pipecat.turns.user_stop.eager_match_policy import EagerMatchPolicy, NormalizedMatch
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy


class TurnAnalyzerEagerUserTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """``TurnAnalyzerUserTurnStopStrategy`` that answers the analyzer's eager end of turn speculatively.

    Events it adds on top of the base strategy: ``on_user_turn_inference_triggered`` with a
    ``UserTurnSpeculation`` when the analyzer reports an eager end of turn, and
    ``on_user_turn_speculation_cancelled`` when that speculation is withdrawn. The committed turn end
    arrives through ``on_user_turn_stopped`` as before, with ``confirms_speculation`` set when the held
    response answers it.
    """

    def __init__(self, *, match_policy: EagerMatchPolicy | None = None, speculation_timeout: float = 5.0, **kwargs):
        """``speculation_timeout``: seconds an eager may go without a commit or a cancel before the
        response is discarded (the analyzer's ``stop_secs`` backstop normally resolves it first)."""
        super().__init__(**kwargs)
        self._match_policy = match_policy or NormalizedMatch()
        self._speculation_timeout = speculation_timeout
        self._speculation: UserTurnSpeculation | None = None
        self._speculation_timeout_task: asyncio.Task | None = None
        self._interim_text = ""

    # --- public state -----------------------------------------------------------------------

    @property
    def speculating(self) -> bool:
        """Whether a speculative response is in flight for the current turn."""
        return self._speculation is not None

    @property
    def match_policy(self) -> EagerMatchPolicy:
        return self._match_policy

    # --- frames -----------------------------------------------------------------------------

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, InterimTranscriptionFrame):
            self._interim_text = frame.text
        elif isinstance(frame, TranscriptionFrame):
            self._interim_text = ""
        return await super().process_frame(frame)

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)
        pop = getattr(self._turn_analyzer, "pop_eager", None)
        eager = bool(pop()) if pop is not None else False
        if state == EndOfTurnState.COMPLETE:
            _, prediction = await self._turn_analyzer.analyze_end_of_turn()
            await self._handle_prediction_result(prediction)
            self._turn_complete = True
            await self._maybe_trigger_user_turn_stopped()
            return
        if eager and self._speculation is None and not self._vad_user_speaking and not self._turn_complete:
            await self._speculate()

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        await super()._handle_vad_user_started_speaking(frame)
        if self._speculation is not None:
            logger.debug(f"{self}: user resumed speaking, withdrawing the eager end of turn")
            await self._withdraw()

    # --- turn end ---------------------------------------------------------------------------

    async def trigger_user_turn_stopped(self, *, enable_user_speaking_frames: bool | None = None):
        speculation = await self._take_speculation()
        if speculation is None:
            await super().trigger_user_turn_stopped(enable_user_speaking_frames=enable_user_speaking_frames)
            return
        if self._match_policy.matches(speculation.text, self._text):
            logger.debug(f"{self}: eager end of turn held, releasing the speculative response")
            await self.trigger_user_turn_finalized(enable_user_speaking_frames=enable_user_speaking_frames, confirms_speculation=True)
            return
        logger.debug(f"{self}: committed transcript differs from the eager one, discarding the speculative response "
                     f"(eager: [{speculation.text}], committed: [{self._text}])")
        await self.trigger_user_turn_speculation_cancelled()
        await super().trigger_user_turn_stopped(enable_user_speaking_frames=enable_user_speaking_frames)

    async def _reset(self):
        speculation = await self._take_speculation()
        await super()._reset()
        self._interim_text = ""
        if speculation is not None:
            # the turn ended some other way (watchdog, interruption, forced stop): nothing else
            # would withdraw the held response
            logger.debug(f"{self}: turn ended with the speculation unresolved, discarding the speculative response")
            await self.trigger_user_turn_speculation_cancelled()

    async def cleanup(self):
        await self._take_speculation()
        await super().cleanup()

    # --- speculation ------------------------------------------------------------------------

    async def _speculate(self):
        speculation = UserTurnSpeculation(text=self._text or self._interim_text)
        self._speculation = speculation
        self._speculation_timeout_task = self.task_manager.create_task(
            self._speculation_timeout_handler(speculation), f"{self}::_speculation_timeout_handler")
        logger.debug(f"{self}: eager end of turn from the analyzer, speculating on [{speculation.text}]")
        await self.trigger_user_turn_inference_triggered(speculation=speculation)

    async def _speculation_timeout_handler(self, speculation: UserTurnSpeculation):
        await asyncio.sleep(self._speculation_timeout)
        self._speculation_timeout_task = None
        if self._speculation is not speculation:
            return
        logger.debug(f"{self}: eager end of turn unresolved after {self._speculation_timeout}s, discarding the speculative response")
        self._speculation = None
        await self.trigger_user_turn_speculation_cancelled()

    async def _withdraw(self):
        if await self._take_speculation() is not None:
            await self.trigger_user_turn_speculation_cancelled()

    async def _take_speculation(self) -> UserTurnSpeculation | None:
        speculation, self._speculation = self._speculation, None
        if self._speculation_timeout_task is not None:
            task, self._speculation_timeout_task = self._speculation_timeout_task, None
            await self.task_manager.cancel_task(task)
        return speculation
