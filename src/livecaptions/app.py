"""Wire config -> source -> sink (terminal or overlay) and run until finished / Ctrl+C."""
from __future__ import annotations

from .capture.cuda import bootstrap_cuda_dlls
from .capture.devices import (
    enumerate_loopbacks,
    name_ordinal,
    print_device_list,
    resolve_loopback,
)
from .capture.wasapi import BlockingWasapiSource, WasapiLoopbackSource
from .capture.wavfile import WavFileSource
from .config import Settings, save_device_choice
from .sources.demo import DemoTranscriptionSource
from .sources.fake import FakeTranscriptionSource
from .store.db import DB_PATH
from .ui.terminal import TerminalUI


def _overrides(args) -> dict:
    o = {}
    if getattr(args, "model", None):
        o["model_name"] = args.model
    if getattr(args, "cpu", False):
        o["device"] = "cpu"
    if getattr(args, "opacity", None) is not None:
        o["overlay_opacity"] = args.opacity
    return o


def _fanout(*sinks):
    """One on_event that feeds several sinks (UI + transcript writer)."""
    def emit(event):
        for sink in sinks:
            sink(event)
    return emit


def _run_terminal(source, settings, *, source_name: str, is_live: bool, extra_sink=None) -> None:
    ui = TerminalUI(source_name=source_name, is_live=is_live,
                    silence_rms_floor=settings.silence_rms_floor,
                    no_blocks_warn_sec=settings.no_blocks_warn_sec,
                    silence_warn_sec=settings.silence_warn_sec,
                    get_dropped=lambda: getattr(source, "dropped_blocks", 0))
    ui.start()
    on_event = _fanout(ui.on_event, extra_sink) if extra_sink else ui.on_event
    source.start(on_event=on_event, monitor=ui.on_block)
    try:
        while not source.finished.wait(0.25):
            pass
    except KeyboardInterrupt:
        ui.message("Stopping...")
    source.stop()
    ui.stop()
    print("Done.")


def _dispatch(source_factory, settings, args, *, source_name: str, is_live: bool,
              extra_sink=None) -> None:
    """`source_factory` is a zero-arg callable that builds the source (it loads the
    Whisper model, which on first run downloads weights). The overlay runs it
    off-thread so the window appears immediately with a status; the terminal path
    builds it eagerly and prints progress to the console."""
    if getattr(args, "overlay", False) or getattr(args, "demo", False) or getattr(args, "screenshot", None):
        from .ui.overlay import run_overlay
        run_overlay(source_factory, settings, source_name=source_name, is_live=is_live,
                    movable=getattr(args, "movable", False),
                    screenshot_path=getattr(args, "screenshot", None),
                    extra_sink=extra_sink)
    else:
        _run_terminal(source_factory(), settings, source_name=source_name, is_live=is_live,
                      extra_sink=extra_sink)


def _run_diarize(args, settings) -> None:
    """Offline post-processing: WAV -> speaker-labeled transcript."""
    from .asr.whisper import load_model   # lazy: pulls faster-whisper (~20s cold)
    from .diarize.assign import format_transcript
    from .diarize.pipeline import diarize_file

    bootstrap_cuda_dlls()
    model = load_model(settings)
    segments, n_speakers = diarize_file(
        args.diarize, model, settings,
        backend=(args.diarizer or settings.diarizer),
        num_speakers=(args.num_speakers if args.num_speakers is not None
                      else settings.diarize_num_speakers))

    text = format_transcript(segments)
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)
    print(f"({len(segments)} segments, {n_speakers} speaker(s))")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Wrote {args.out}")


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _run_name_speakers(args) -> None:
    """Suggest real names for SPEAKER_N labels using Claude.

    Two gates, because this is the only feature that leaves the machine: consent
    before the transcript is sent, and confirmation before any rename is written.
    """
    from .store.db import connect
    from .store.naming import build_transcript, propose_names, session_labels
    from .store.search import rename_speaker

    if not args.session:
        raise SystemExit("--name-speakers needs --session N (see --sessions)")

    conn = connect()
    labels = session_labels(conn, args.session)
    if not labels:
        raise SystemExit(f"Session {args.session} has no speaker labels to name.")

    transcript, truncated = build_transcript(conn, args.session)
    if not transcript:
        raise SystemExit(f"Session {args.session} has no text.")

    # Gate 1: consent to send. Everything else in this app is local; be explicit.
    print(f"Session {args.session}: {len(labels)} speaker(s) - {', '.join(labels)}")
    print(f"This will send {len(transcript):,} characters of transcript text to the "
          f"Anthropic API ({args.name_model}).")
    if truncated:
        print("(Transcript is long - only the earlier part will be sent.)")
    if not args.yes and not _confirm("Send it?"):
        print("Cancelled. Nothing was sent.")
        return

    print("Asking the model...")
    try:
        proposals = propose_names(transcript, labels, model=args.name_model)
    except Exception as e:
        raise SystemExit(f"Naming failed: {type(e).__name__}: {e}")

    # Gate 2: confirm each rename, with the evidence the model cited.
    applied = 0
    for p in proposals:
        print()
        if not p.name:
            print(f"  {p.label}: no name found - {p.evidence}")
            continue
        print(f"  {p.label} -> {p.name}   (confidence: {p.confidence})")
        print(f"    evidence: {p.evidence}")
        if p.confidence == "high" and args.apply_high:
            ok = True
        else:
            ok = _confirm(f"    Rename {p.label} to {p.name}?")
        if ok:
            n = rename_speaker(conn, p.label, p.name, session_id=args.session)
            print(f"    Renamed {n} line(s).")
            applied += 1
        else:
            print("    Skipped.")

    print(f"\n{applied} name(s) applied. Reversible: "
          f"--rename-speaker NEW=OLD --session {args.session}")


def _run_store_command(args) -> bool:
    """Handle the read-only transcript-store commands. True if one ran."""
    from .store.db import connect
    from .store.export import export
    from .store.search import recent_sessions, rename_speaker, search

    if args.sessions:
        conn = connect()
        rows = recent_sessions(conn)
        if not rows:
            print("No saved sessions yet.")
        for r in rows:
            title = f"  {r['title']}" if r["title"] else ""
            print(f"[{r['id']:>3}] {r['started_at']}  {r['utterances']:>4} lines, "
                  f"{r['speakers']} speaker(s)  ({r['source'] or '?'}){title}")
        return True

    if args.search:
        conn = connect()
        hits = search(conn, args.search, speaker=args.speaker, since=args.since)
        if not hits:
            print("No matches.")
        for h in hits:
            who = f"{h.speaker}: " if h.speaker else ""
            print(f"[s{h.session_id} @{h.t_start:7.1f}s] {h.wall_clock}  {who}{h.snippet}")
        print(f"({len(hits)} match(es))")
        return True

    if args.export:
        if not args.session:
            raise SystemExit("--export needs --session N (see --sessions)")
        conn = connect()
        text = export(conn, args.session, args.export)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Wrote {args.out}")
        else:
            print(text)
        return True

    if getattr(args, "name_speakers", False):
        _run_name_speakers(args)
        return True

    if args.rename_speaker:
        if "=" not in args.rename_speaker:
            raise SystemExit("--rename-speaker takes OLD=NEW (e.g. SPEAKER_00=Sarah)")
        old, new = args.rename_speaker.split("=", 1)
        conn = connect()
        n = rename_speaker(conn, old.strip(), new.strip(), session_id=args.session)
        print(f"Renamed {n} line(s): {old.strip()} -> {new.strip()}   (reversible: swap the two)")
        return True

    return False


def run(args) -> None:
    import pyaudiowpatch as pyaudio

    if _run_store_command(args):
        return

    if getattr(args, "download_models", False):
        from .diarize.models import download_sherpa_models
        download_sherpa_models()
        return

    if args.list_devices:
        p = pyaudio.PyAudio()
        try:
            print_device_list(p)
        finally:
            p.terminate()
        return

    settings = Settings(**_overrides(args))

    if getattr(args, "diarize", None):
        _run_diarize(args, settings)
        return

    if getattr(args, "screenshot", None):
        args.demo = True   # screenshot renders canned content; no audio/model needed

    # --- Demo / Fake sources: no audio device, no GPU. ---
    if getattr(args, "demo", False):
        print("Demo overlay (canned partials + finals, no audio/GPU).")
        _dispatch(lambda: DemoTranscriptionSource(loop=getattr(args, "loop", False)),
                  settings, args, source_name="demo", is_live=False)
        return
    if getattr(args, "fake", False):
        print("Fake source (canned captions, no audio/GPU).")
        _dispatch(lambda: FakeTranscriptionSource(), settings, args, source_name="fake", is_live=False)
        return

    # --- Build the audio source first (fail fast on a bad file/device). ---
    if args.wav:
        audio = WavFileSource(args.wav, block_sec=settings.block_sec, paced=not args.wav_fast)
    else:
        p = pyaudio.PyAudio()
        try:
            dev = resolve_loopback(
                p, index=args.loopback_index, device_substr=args.device, pick=args.pick,
                saved_name=settings.loopback_name, saved_ordinal=settings.loopback_ordinal)
            if args.device or args.pick:
                save_device_choice(dev["name"], name_ordinal(enumerate_loopbacks(p), dev))
        finally:
            p.terminate()
        cls = BlockingWasapiSource if args.blocking else WasapiLoopbackSource
        audio = cls(dev, block_sec=settings.block_sec)

    live_diarize = getattr(args, "diarize_live", False) or getattr(settings, "speaker_colors", False)
    streaming = getattr(args, "streaming", False) or live_diarize

    def build_source():
        """Deferred so the overlay can show a 'loading' status while this runs.
        Imports faster-whisper here (not at module load) so importing app is fast
        and the overlay appears before the ~20 s cold ML import, not after it."""
        from .asr.whisper import load_model
        from .sources.local import LocalTranscriptionSource
        bootstrap_cuda_dlls()
        model = load_model(settings)
        if streaming:
            from .sources.streaming_local import StreamingTranscriptionSource
            return StreamingTranscriptionSource(audio, model, settings, source_id="loopback",
                                                diarize=live_diarize)
        return LocalTranscriptionSource(audio, model, settings, source_id="loopback")

    if audio.is_live:
        print(f"Capturing: {audio.name}  ({audio.rate} Hz, index {getattr(audio, 'index', '?')})")
        print("Play some audio (a call, a video...) and speak. Ctrl+C to stop.\n")
    else:
        print(f"Source: {audio.name}  ({audio.rate} Hz)")
        print("Replaying WAV...\n")

    writer = None
    if not getattr(args, "no_save", False):
        from .store.writer import TranscriptWriter
        writer = TranscriptWriter(source=audio.name)
        writer.start()

    try:
        _dispatch(build_source, settings, args, source_name=audio.name, is_live=audio.is_live,
                  extra_sink=(writer.on_event if writer else None))
    finally:
        if writer is not None:
            writer.stop()
            if writer.count:
                print(f"Saved {writer.count} line(s) to session {writer.session_id} "
                      f"({DB_PATH})\n  search with:  python -m livecaptions --search \"...\"")
