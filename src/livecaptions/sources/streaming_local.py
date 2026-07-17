"""Streaming local transcription: continuous partial + final captions.

Instead of waiting for silence to segment (LocalTranscriptionSource), this feeds
a rolling buffer to Whisper repeatedly and commits words with LocalAgreement-2,
so captions appear and stabilise during unbroken speech. Committed words extend
the live (partial) line; a line finalises on a VAD pause, sentence punctuation,
or a max length.

Backpressure is "latest wins": audio always accumulates in the OnlineASR buffer;
if decoding is slower than real time, the next pass simply covers more audio
(self-coalescing) rather than queueing stale work.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import List, Optional

import numpy as np
import soxr

from ..asr.segmenter import Segmenter
from ..asr.streaming import WHISPER_SR, OnlineASR
from ..asr.vad import SpeechGate
from ..capture.base import AudioSource
from ..events import TranscriptEvent
from ..util import drop_oldest_put
from .base import EventCallback, TranscriptionSource

# Common Whisper hallucinations on silence/music/ads — dropped from FINALS.
JUNK_FINALS = {
    "", ".", "you", "so", "the", "thank you.", "thank you", "thanks for watching!",
    "thanks for watching.", "thank you for watching.", "thank you for listening.",
    "thanks for listening.", "thank you very much.", "bye.", "bye bye.", "okay.",
    "please subscribe.", "please subscribe to my channel.", "subtitles by the amara.org community",
    "i'm sorry.", "thank you so much.", "thank you all so much.",
}


#: minimum words in a line before a diarizer speaker-change is allowed to break it
#: (shorter fragments at a turn boundary are usually boundary lag, not a real turn)
MIN_WORDS_FOR_SPEAKER_CUT = 3


def _clean(words: List) -> str:
    # Whisper word tokens carry their own leading spaces; normalise to single spaces.
    return " ".join(w[2].strip() for w in words if w[2].strip())


class StreamingTranscriptionSource(TranscriptionSource):
    def __init__(self, audio: AudioSource, model, settings, *, source_id: str = "loopback",
                 diarize: bool = False):
        self.source_id = source_id
        self._audio = audio
        self._s = settings
        self._diarizer = None
        if diarize:
            from ..diarize.streaming_sortformer import StreamingSortformer
            print("Loading live diarizer (Streaming Sortformer, CPU)...")
            self._diarizer = StreamingSortformer(
                device=settings.diarize_live_device,
                threshold=settings.diarize_live_threshold)
            print("Live diarizer ready (max 4 speakers; best-effort on a mixed stream).")
        self._online = OnlineASR(model, language=settings.language, beam_size=settings.beam_size,
                                 max_buffer_sec=settings.stream_max_buffer_sec)
        self._vad = SpeechGate(threshold=settings.stream_vad_threshold)
        self._in_rate = audio.rate
        self._resampler: Optional[soxr.ResampleStream] = None
        self._block_q: "queue.Queue" = queue.Queue(maxsize=max(1, int(20 / settings.block_sec)))
        self._on_event: Optional[EventCallback] = None
        self._monitor = None
        self._thread = None
        self._audio_error: Optional[BaseException] = None
        self._stream_time = 0.0        # total 16k audio fed — the shared global clock
        self.dropped_blocks = 0
        self.finished = threading.Event()

    def start(self, on_event: EventCallback, monitor=None) -> None:
        self._on_event = on_event
        self._monitor = monitor
        self._resampler = soxr.ResampleStream(self._in_rate, WHISPER_SR, 1, dtype="float32")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._audio.start(self._enqueue_block, self._on_audio_end)

    def _enqueue_block(self, block) -> None:
        if drop_oldest_put(self._block_q, block):
            self.dropped_blocks += 1

    def _on_audio_end(self, error) -> None:
        self._audio_error = error
        self._block_q.put(None)

    def _emit(self, words: List, is_final: bool, speaker: Optional[str] = None) -> None:
        text = _clean(words)
        if is_final and (not text or text.lower() in JUNK_FINALS):
            return
        if not text:
            return
        t_start, t_end = words[0][0], words[-1][1]
        if speaker is None and self._diarizer is not None:
            speaker = self._diarizer.speaker_at(t_start, t_end)
        if os.environ.get("LC_STREAM_DEBUG"):
            who = f"{speaker} " if speaker else ""
            print(f"  [{time.strftime('%M:%S')}] {'FINAL' if is_final else 'part '} "
                  f"{who}{text!r}", flush=True)
        self._on_event(TranscriptEvent(
            text=text, source=self.source_id, speaker=speaker,
            t_start=t_start, t_end=t_end, is_final=is_final))

    def _word_speaker(self, word) -> Optional[str]:
        if self._diarizer is None:
            return None
        return self._diarizer.speaker_at(word[0], word[1])

    def _run(self) -> None:
        pending: List = []              # committed words not yet emitted as a final line
        line_speaker: Optional[str] = None   # whose line `pending` is
        new_audio = 0.0                 # seconds of 16k audio since the last decode
        ended = False

        def cut_line():
            nonlocal pending, line_speaker
            self._emit(pending, is_final=True, speaker=line_speaker)
            pending = []
            line_speaker = None
            self._online.trim_to_committed()   # drop finalised audio, keep the tail

        while True:
            try:
                block = self._block_q.get(timeout=0.2)
                batch = [] if block is None else [block]
                if block is None:
                    ended = True
                else:
                    while True:
                        try:
                            b = self._block_q.get_nowait()
                        except queue.Empty:
                            break
                        if b is None:
                            ended = True
                        else:
                            batch.append(b)
            except queue.Empty:
                batch = []

            for b in batch:
                if self._monitor is not None:
                    self._monitor(Segmenter.block_rms(b.samples))
                chunk = self._resampler.resample_chunk(b.samples.astype(np.float32))
                if chunk.size:
                    self._online.insert_audio(chunk)
                    if self._diarizer is not None:
                        self._diarizer.feed(chunk)      # same audio, same clock
                    self._stream_time += chunk.size / WHISPER_SR
                    new_audio += chunk.size / WHISPER_SR

            if ended:
                break
            if new_audio < self._s.stream_process_interval:
                continue
            new_audio = 0.0

            buf = self._online.buffer
            if not self._vad.has_speech(buf):
                if pending:
                    cut_line()          # end of an utterance: commit the sentence
                # drop the silence, but keep the global clock so word timestamps
                # stay aligned with the diarizer's speaker timeline
                self._online.reset(self._stream_time)
                continue

            committed, tail = self._online.process()
            for w in committed:
                spk = self._word_speaker(w)
                # A speaker change ends the line — otherwise one line merges the tail
                # of one turn with the start of the next and gets mislabelled. But
                # diarization boundaries lag speech onset slightly, so a 1-2 word
                # fragment at a turn boundary is usually just misattributed: absorb
                # it into the new speaker's line instead of emitting a stray line.
                if pending and spk and line_speaker and spk != line_speaker:
                    if len(pending) >= MIN_WORDS_FOR_SPEAKER_CUT:
                        cut_line()
                    else:
                        line_speaker = spk
                if line_speaker is None:
                    line_speaker = spk
                pending.append(w)
            self._emit(pending + tail, is_final=False, speaker=line_speaker)   # live partial

            trailing = self._vad.trailing_silence_sec(buf)
            dur = (pending[-1][1] - pending[0][0]) if pending else 0.0
            ends_sentence = bool(pending) and pending[-1][2].strip().endswith((".", "?", "!"))
            # break a line on a completed sentence, a long run-on, or a pause —
            # WITHOUT resetting the ASR, so the next sentence's audio isn't lost
            if pending and (ends_sentence
                            or dur >= self._s.stream_max_line_sec
                            or trailing >= self._s.stream_end_silence_sec):
                cut_line()

        # end of stream: flush the resampler, decode the remainder, finalise
        chunk = self._resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        if chunk.size:
            self._online.insert_audio(chunk)
        if self._online.buffer.size and self._vad.has_speech(self._online.buffer):
            committed, tail = self._online.process()
            pending.extend(committed)
            pending.extend(tail)        # accept the unconfirmed tail at end-of-stream
        if pending:
            cut_line()
        if self._audio_error is not None:
            print(f"Audio capture failed: {self._audio_error}")
        self.finished.set()

    def stop(self) -> None:
        self._audio.stop()
        self._block_q.put(None)
