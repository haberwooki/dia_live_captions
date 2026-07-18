"""CLI entrypoint:  python -m livecaptions  (or the `livecaptions` script)."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="livecaptions",
        description="Live captions from Windows system audio (local GPU transcription).")
    ap.add_argument("--list-devices", action="store_true",
                    help="list loopback capture devices and exit")
    ap.add_argument("--device", metavar="SUBSTR",
                    help="use the loopback device whose name contains SUBSTR (remembered)")
    ap.add_argument("--loopback-index", type=int, metavar="N",
                    help="use this exact WASAPI loopback device index")
    ap.add_argument("--pick", action="store_true",
                    help="choose a loopback device interactively (remembered)")
    ap.add_argument("--wav", metavar="PATH",
                    help="transcribe a WAV file instead of live capture (deterministic)")
    ap.add_argument("--wav-fast", action="store_true",
                    help="with --wav, feed as fast as possible (no real-time pacing)")
    ap.add_argument("--blocking", action="store_true",
                    help="use the blocking-read capture fallback instead of callback mode")
    ap.add_argument("--fake", action="store_true",
                    help="run a fake source (canned captions, no audio/GPU) to test wiring")
    ap.add_argument("--model", metavar="NAME",
                    help="override the Whisper model (small / medium / large-v3)")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--streaming", action="store_true",
                    help="streaming mode: continuous partial+final captions (LocalAgreement-2)")
    ap.add_argument("--diarize-live", action="store_true",
                    help="live speaker colours via Streaming Sortformer (implies --streaming; "
                         "max 4 speakers, best-effort on a mixed stream)")

    # offline diarization ("who is talking") post-processing pass
    ap.add_argument("--diarize", metavar="WAV",
                    help="offline pass: transcribe WAV and label who said what")
    ap.add_argument("--diarizer", choices=["auto", "pyannote", "sherpa"], default=None,
                    help="diarization backend (auto: pyannote if an HF token is available)")
    ap.add_argument("--num-speakers", type=int, metavar="N",
                    help="tell the diarizer how many speakers to expect (default: infer)")
    ap.add_argument("--out", metavar="PATH", help="write the labeled transcript to a file")
    ap.add_argument("--download-models", action="store_true",
                    help="fetch the sherpa-onnx diarization models and exit")

    # saved & searchable transcripts
    ap.add_argument("--no-save", action="store_true",
                    help="don't save this session's captions to the transcript store")
    ap.add_argument("--search", metavar="QUERY",
                    help="search saved transcripts (FTS5: \"quoted phrases\", AND/OR/NOT)")
    ap.add_argument("--sessions", action="store_true", help="list recent saved sessions")
    ap.add_argument("--speaker", metavar="NAME", help="with --search: filter by speaker")
    ap.add_argument("--since", metavar="ISO_DATE", help="with --search: only after this date")
    ap.add_argument("--export", metavar="FMT", choices=["srt", "vtt", "jsonl", "md"],
                    help="export a saved session (use --session N)")
    ap.add_argument("--session", type=int, metavar="N", help="session id for --export")
    ap.add_argument("--rename-speaker", metavar="OLD=NEW",
                    help="rename a speaker in saved transcripts (reversible)")
    ap.add_argument("--name-speakers", action="store_true",
                    help="use Claude to suggest real names for SPEAKER_N in a session "
                         "(use --session N). SENDS THE TRANSCRIPT TO THE ANTHROPIC API; "
                         "asks before sending and before applying anything")
    ap.add_argument("--name-model", metavar="ID", default="claude-opus-4-8",
                    help="model for --name-speakers (default: claude-opus-4-8)")
    ap.add_argument("--yes", action="store_true",
                    help="with --name-speakers: skip the send prompt (still confirms "
                         "each rename unless --apply-high is also given)")
    ap.add_argument("--apply-high", action="store_true",
                    help="with --name-speakers: auto-apply high-confidence names "
                         "without asking per name (still reversible)")

    # overlay (M2)
    ap.add_argument("--overlay", action="store_true",
                    help="show captions in an always-on-top overlay instead of the terminal")
    ap.add_argument("--demo", action="store_true",
                    help="overlay demo: canned partials+finals, no audio/GPU (implies --overlay)")
    ap.add_argument("--settings", action="store_true",
                    help="open the settings window (device, size, colour, model) and exit; "
                         "no capture")
    ap.add_argument("--movable", action="store_true",
                    help="overlay: disable click-through so you can drag it to reposition")
    ap.add_argument("--opacity", type=float, metavar="F",
                    help="overlay opacity 0.0-1.0")
    ap.add_argument("--loop", action="store_true", help="with --demo, loop forever")
    ap.add_argument("--screenshot", metavar="PATH",
                    help="render the overlay to a PNG and exit (for testing)")
    return ap


def main() -> None:
    # Before anything imports huggingface_hub: put the model cache in an
    # app-owned dir (packaged build) and capture stdio to a log for the
    # windowed EXE. No-op for the important parts when running from source.
    from .runtime import configure_runtime
    configure_runtime()

    args = build_parser().parse_args()

    # --settings only needs the config window — route it before importing app,
    # which pulls in the ~20 s cold faster-whisper stack it doesn't need.
    if args.settings:
        from .config import Settings
        from .ui.settings import run_settings
        run_settings(Settings(), screenshot_path=args.screenshot)
        return

    from .app import run
    run(args)


if __name__ == "__main__":
    main()
