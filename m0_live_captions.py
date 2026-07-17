"""
M0.1 spike — live captions from Windows system audio, local + GPU.

Captures whatever is playing out of your speakers (Discord, Teams, Meet,
a YouTube video — anything) via WASAPI loopback, and prints live
transcription using faster-whisper on your GPU (with automatic CPU fallback).

This is the "prove the risky bits" milestone: no overlay, no speaker labels
yet. It exists to prove capture -> text works, to be diagnosable rather than a
silent black box, and to measure latency. It detects your hardware at runtime
and adapts (any Windows PC, GPU or CPU).

Usage:
  python m0_live_captions.py                  # live capture from the default output's loopback
  python m0_live_captions.py --list-devices   # list loopback capture devices and exit
  python m0_live_captions.py --device DELL     # pick a loopback whose name contains "DELL" (and remember it)
  python m0_live_captions.py --pick            # choose a loopback interactively (and remember it)
  python m0_live_captions.py --wav clip.wav    # transcribe a WAV file (deterministic, no capture device)
Stop: Ctrl+C
"""

import os
import sys
import json
import math
import time
import wave
import queue
import argparse
import threading

import numpy as np

try:
    import soxr
except ImportError:
    print("Missing dependency 'soxr'. Run: pip install soxr")
    sys.exit(1)

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("Missing dependency 'PyAudioWPatch'. Run: pip install PyAudioWPatch")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Missing dependency 'faster-whisper'. Run: pip install faster-whisper")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config — tweak these
# ---------------------------------------------------------------------------
MODEL_NAME    = "medium"      # small / medium / large-v3 ; medium is a good 8 GB-GPU balance
DEVICE        = "auto"        # "auto" = GPU if it works, else CPU; force with "cuda"/"cpu"
GPU_COMPUTE   = "float16"     # compute type on GPU
CPU_COMPUTE   = "int8"        # compute type on the CPU fallback
LANGUAGE      = "en"          # set to None for auto multilingual (slower, can flip languages)
BEAM_SIZE     = 1            # 1 = fast/greedy (good for live); 5 = more accurate/slower

LOOPBACK_INDEX = None        # default for --loopback-index; None = auto-detect

WHISPER_SR    = 16000        # faster-whisper wants 16 kHz mono
BLOCK_SEC     = 0.1          # audio read granularity
SILENCE_RMS   = 350          # int16 RMS below this = "silence" (raise if it never ends an utterance)
SILENCE_RMS_FLOOR = 5.0      # below this = effectively digital silence (for the health watchdog)
END_SILENCE   = 0.6          # seconds of trailing silence that ends an utterance
MIN_UTT_SEC   = 0.4          # ignore blips shorter than this
MAX_UTT_SEC   = 12.0         # force a flush after this long even without a pause

NO_BLOCKS_WARN_SEC = 4.0     # no audio blocks at all for this long -> idle/phantom-endpoint warning
SILENCE_WARN_SEC   = 8.0     # blocks arriving but ~silence for this long -> wrong/muted-output warning
# ---------------------------------------------------------------------------


QUEUE_MAX = max(1, int(20 / BLOCK_SEC))   # ~20 s of audio; bounds memory if inference lags
audio_q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=QUEUE_MAX)
stop_flag = threading.Event()
capture_error: list = []   # a source records a fatal error here before pushing its None sentinel
dropped_blocks = 0         # audio blocks dropped because the consumer fell behind


def enqueue(item):
    """Put a block on the queue without ever blocking a producer (esp. the audio
    callback). If the consumer has fallen behind and the queue is full, drop the
    OLDEST audio block to make room — bounded memory, at the cost of a gap."""
    global dropped_blocks
    try:
        audio_q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        audio_q.get_nowait()
        dropped_blocks += 1
    except queue.Empty:
        pass
    try:
        audio_q.put_nowait(item)
    except queue.Full:
        pass


def _config_dir():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "live-captions")


CONFIG_FILE = os.path.join(_config_dir(), "config.json")


# ---------------------------------------------------------------------------
# CUDA / model loading
# ---------------------------------------------------------------------------
def bootstrap_cuda_dlls():
    """Make CUDA libraries from the nvidia-*-cu12 pip wheels loadable.

    faster-whisper's CTranslate2 backend loads cuBLAS/cuDNN dynamically, but on
    Windows its internal loader does not search the nvidia wheel directories.
    We add those dirs to the DLL search path and preload the libraries by name,
    so a later LoadLibrary("cublas64_12.dll") resolves the already-resident
    module. Best-effort and silent: on CPU-only machines the wheels aren't
    installed and we just run on the CPU. No-op off Windows.
    """
    if os.name != "nt":
        return
    import glob, sysconfig, ctypes
    site = sysconfig.get_paths()["purelib"]
    for d in glob.glob(os.path.join(site, "nvidia", "*", "bin")):
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    for name in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"):
        try:
            ctypes.WinDLL(name)
        except OSError:
            pass


def load_model():
    """Load Whisper, preferring the GPU and falling back to the CPU.

    GPU (cuBLAS/cuDNN) problems often surface only at the FIRST inference rather
    than at load time, so each candidate is validated with a tiny self-test
    transcription before we commit to it.
    """
    if DEVICE == "cpu":
        candidates = [("cpu", CPU_COMPUTE)]
    elif DEVICE == "cuda":
        candidates = [("cuda", GPU_COMPUTE)]
    else:  # "auto"
        candidates = [("cuda", GPU_COMPUTE), ("cpu", CPU_COMPUTE)]

    last_err = None
    for device, compute in candidates:
        try:
            print(f"Loading faster-whisper '{MODEL_NAME}' on {device} ({compute})...")
            t0 = time.time()
            model = WhisperModel(MODEL_NAME, device=device, compute_type=compute)
            # force the encoder to run so a broken GPU path fails HERE, not mid-stream
            list(model.transcribe(np.zeros(WHISPER_SR, dtype=np.float32),
                                  beam_size=1, vad_filter=False)[0])
            print(f"Model ready on {device} in {time.time() - t0:.1f}s.")
            return model
        except Exception as e:
            last_err = e
            print(f"  {device} path unavailable: {e}")
            if device == "cuda":
                print("  Falling back to CPU (slower). For the GPU path install the CUDA "
                      "wheels: pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12")

    print(f"\nCould not load the model on any device. Last error: {last_err}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Device enumeration / selection
# ---------------------------------------------------------------------------
def enumerate_loopbacks(p):
    """All WASAPI loopback capture devices, in enumeration order."""
    return list(p.get_loopback_device_info_generator())


def _name_ordinal(loopbacks, dev):
    """Position of `dev` among loopbacks that share its exact name (0-based)."""
    same = [lb for lb in loopbacks if lb["name"] == dev["name"]]
    for i, lb in enumerate(same):
        if lb["index"] == dev["index"]:
            return i
    return 0


def default_loopback(p):
    """The loopback for the default output, disambiguating duplicate names.

    Machines with several identically-named outputs (e.g. two identical
    monitors) expose several identically-named loopbacks; a naive first-name
    match can grab a silent/idle endpoint. Map the default output's position
    among same-named render devices to the same position in the loopback list.
    """
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if default_speakers.get("isLoopbackDevice", False):
        return default_speakers

    name = default_speakers["name"]
    matches = [lb for lb in p.get_loopback_device_info_generator() if name in lb["name"]]
    if not matches:
        raise RuntimeError(
            "No loopback device found for the default output. "
            "Make sure you're on Windows with WASAPI."
        )
    if len(matches) == 1:
        return matches[0]

    same_named = []
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if (d["name"] == name and d.get("hostApi") == wasapi["index"]
                and d.get("maxOutputChannels", 0) > 0
                and not d.get("isLoopbackDevice", False)):
            same_named.append(d["index"])
    try:
        pos = same_named.index(default_speakers["index"])
    except ValueError:
        pos = 0
    chosen = matches[min(pos, len(matches) - 1)]
    print(f"(multiple '{name}' outputs found; using loopback index {chosen['index']} - "
          f"override with --device / --loopback-index)")
    return chosen


def save_device_choice(loopbacks, dev):
    """Remember a chosen device by its loopback name + ordinal (indices are volatile)."""
    data = {"loopback_name": dev["name"], "loopback_ordinal": _name_ordinal(loopbacks, dev)}
    try:
        os.makedirs(_config_dir(), exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError as e:
        print(f"(could not save device choice: {e})")


def load_saved_device(loopbacks):
    """Resolve a previously-saved choice to a current device, or None."""
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):   # valid JSON but not our object -> ignore
        return None
    name = data.get("loopback_name")
    ordinal = data.get("loopback_ordinal", 0)
    same = [lb for lb in loopbacks if lb["name"] == name]
    if not same:
        return None
    return same[min(ordinal, len(same) - 1)]


def interactive_pick(loopbacks):
    print("Loopback capture devices:")
    for i, lb in enumerate(loopbacks):
        print(f"  [{i}] index {lb['index']}: {lb['name']} "
              f"({int(lb['defaultSampleRate'])} Hz, {lb['maxInputChannels']} ch)")
    while True:
        try:
            choice = int(input("Pick a number: "))
            if not 0 <= choice < len(loopbacks):   # reject negatives (they'd wrap) and out-of-range
                raise IndexError
            return loopbacks[choice]
        except (ValueError, IndexError):
            print("  invalid choice, try again")
        except EOFError:
            raise SystemExit("--pick needs an interactive terminal")


def print_device_list(p):
    loopbacks = enumerate_loopbacks(p)
    # Resolving the default output can fail (no default endpoint, headless/RDP).
    # This command must still list the loopbacks it already has -- it's the
    # thing you run WHEN device resolution is misbehaving.
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    try:
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        default_lb = default_loopback(p)
    except Exception:
        default_out = None
        default_lb = None
    if default_out is not None:
        print(f"Default output: {default_out['name']} (index {default_out['index']})\n")
    else:
        print("Default output: (none detected)\n")
    print("Loopback capture devices:")
    for lb in loopbacks:
        mark = "   <- default (auto)" if (default_lb and lb["index"] == default_lb["index"]) else ""
        print(f"  index {lb['index']}: {lb['name']} "
              f"({int(lb['defaultSampleRate'])} Hz, {lb['maxInputChannels']} ch){mark}")


def resolve_loopback(p, args):
    """Pick the loopback to capture: index > --device > --pick > saved > default."""
    loopbacks = enumerate_loopbacks(p)
    if args.loopback_index is not None:
        try:
            dev = p.get_device_info_by_index(args.loopback_index)
        except (OSError, ValueError):
            raise SystemExit(f"No device at --loopback-index {args.loopback_index}. "
                             f"Try --list-devices.")
        if not dev.get("isLoopbackDevice", False):
            raise SystemExit(f"Index {args.loopback_index} ('{dev['name']}') is not a loopback "
                             f"capture device. Try --list-devices.")
        return dev
    if args.device:
        matches = [lb for lb in loopbacks if args.device.lower() in lb["name"].lower()]
        if not matches:
            raise SystemExit(f"No loopback device name contains '{args.device}'. "
                             f"Try --list-devices.")
        if len(matches) > 1:
            # Ambiguous (e.g. two identical monitors): prefer the active default
            # output's loopback if it's among the matches, else the first.
            try:
                dfl = default_loopback(p)
            except Exception:   # any failure to resolve the default -> just use matches[0]
                dfl = None
            chosen = dfl if (dfl and any(m["index"] == dfl["index"] for m in matches)) else matches[0]
            print(f"(--device '{args.device}' matched {len(matches)} devices; "
                  f"using index {chosen['index']} - use --loopback-index to force another)")
        else:
            chosen = matches[0]
        save_device_choice(loopbacks, chosen)
        return chosen
    if args.pick:
        dev = interactive_pick(loopbacks)
        save_device_choice(loopbacks, dev)
        return dev
    saved = load_saved_device(loopbacks)
    if saved is not None:
        return saved
    try:
        return default_loopback(p)
    except (RuntimeError, OSError):
        raise SystemExit(
            "No usable audio output/loopback device found. Make sure something is set as your "
            "Windows default playback device (a sleeping monitor or unplugged output can remove "
            "it). Run --list-devices to see what's available.")


# ---------------------------------------------------------------------------
# Audio sources — each pushes native-rate int16 MONO blocks (bytes) onto
# audio_q, then a None sentinel at end/error (with capture_error set on error).
# ---------------------------------------------------------------------------
class LoopbackCapture:
    """WASAPI loopback capture in callback (non-blocking) mode.

    Callback mode is the primary path: a callback that stops firing is
    detectable by the consumer's watchdog, and the stream tears down cleanly
    (unlike a thread wedged inside a blocking read()).
    """
    is_live = True

    def __init__(self, dev):
        self.name = dev["name"]
        self.index = dev["index"]
        self.rate = int(dev["defaultSampleRate"])
        self.channels = int(dev["maxInputChannels"])
        self._pa = None
        self._stream = None

    def start(self):
        self._pa = pyaudio.PyAudio()
        ch = self.channels

        def _cb(in_data, frame_count, time_info, status):
            try:
                if ch > 1:
                    s = np.frombuffer(in_data, dtype=np.int16).reshape(-1, ch)
                    enqueue(s.mean(axis=1).astype(np.int16).tobytes())
                else:
                    enqueue(in_data)
            except Exception as e:  # never let the audio callback die silently
                capture_error.append(e)
                enqueue(None)
                return (None, pyaudio.paComplete)
            return (None, pyaudio.paContinue)

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16, channels=ch, rate=self.rate,
                frames_per_buffer=int(self.rate * BLOCK_SEC), input=True,
                input_device_index=self.index, stream_callback=_cb,
            )
            self._stream.start_stream()
        except Exception as e:
            capture_error.append(e)
            enqueue(None)

    def stop(self):
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        finally:
            if self._pa is not None:
                self._pa.terminate()


class BlockingLoopbackCapture:
    """Blocking-read fallback (documented interim; use --blocking).

    Kept only in case callback mode misbehaves on some driver. Note the known
    limitation this whole milestone exists to fix: if the endpoint delivers
    nothing, stream.read() blocks inside C and this thread cannot be torn down
    cleanly — the watchdog still reports it, but the thread leaks until exit.
    """
    is_live = True

    def __init__(self, dev):
        self.name = dev["name"]
        self.index = dev["index"]
        self.rate = int(dev["defaultSampleRate"])
        self.channels = int(dev["maxInputChannels"])
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        pa = pyaudio.PyAudio()
        ch = self.channels
        frames = int(self.rate * BLOCK_SEC)
        try:
            stream = pa.open(format=pyaudio.paInt16, channels=ch, rate=self.rate,
                             frames_per_buffer=frames, input=True,
                             input_device_index=self.index)
        except Exception as e:
            capture_error.append(e)
            enqueue(None)
            pa.terminate()
            return
        try:
            while not stop_flag.is_set():
                data = stream.read(frames, exception_on_overflow=False)
                s = np.frombuffer(data, dtype=np.int16)
                if ch > 1:
                    s = s.reshape(-1, ch).mean(axis=1).astype(np.int16)
                enqueue(s.tobytes())
        except Exception as e:
            capture_error.append(e)
            enqueue(None)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def stop(self):
        stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


class WavFileAudioSource:
    """Replay a 16-bit PCM WAV through the same pipeline (deterministic test rig)."""
    is_live = False

    def __init__(self, path, paced=True):
        try:
            with wave.open(path, "rb") as w:
                self.rate = w.getframerate()
                self._channels = w.getnchannels()
                sampwidth = w.getsampwidth()
                self._raw = w.readframes(w.getnframes())
        except (OSError, wave.Error, EOFError) as e:
            raise SystemExit(f"{path}: cannot read as a WAV file ({e})")
        if sampwidth != 2:
            raise SystemExit(f"{path}: need a 16-bit PCM WAV (sample width {sampwidth} bytes)")
        self.name = f"WAV:{os.path.basename(path)}"
        self._paced = paced
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._feed, daemon=True)
        self._thread.start()

    def _feed(self):
        try:
            data = np.frombuffer(self._raw, dtype=np.int16)
            if self._channels > 1:
                # trim to whole frames first (a truncated WAV may not be a
                # whole number of interleaved samples)
                usable = (len(data) // self._channels) * self._channels
                data = data[:usable].reshape(-1, self._channels).mean(axis=1).astype(np.int16)
            block = max(1, int(self.rate * BLOCK_SEC))
            for i in range(0, len(data), block):
                if stop_flag.is_set():
                    break
                enqueue(data[i:i + block].tobytes())
                if self._paced:
                    time.sleep(BLOCK_SEC)
        except Exception as e:
            capture_error.append(e)
        finally:
            enqueue(None)   # always release the consumer, even on error

    def stop(self):
        stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Terminal status line (live VU meter + messages that don't clobber it)
# ---------------------------------------------------------------------------
class StatusLine:
    """A single rewritable status line. On a non-tty (redirected), the live
    line is suppressed so logs stay clean; messages always print."""

    def __init__(self):
        self._tty = sys.stdout.isatty()
        self._len = 0

    def status(self, text):
        if not self._tty:
            return
        pad = max(0, self._len - len(text))
        sys.stdout.write("\r" + text + " " * pad)
        sys.stdout.flush()
        self._len = len(text)

    def message(self, text):
        if self._tty and self._len:
            sys.stdout.write("\r" + " " * self._len + "\r")
            self._len = 0
        print(text, flush=True)


def _vu_meter(rms, name):
    dbfs = 20 * math.log10(max(rms, 1e-6) / 32768.0)
    width = 28
    level = int(max(0, min(width, (dbfs + 60) / 60 * width)))
    bar = "#" * level + "-" * (width - level)
    return f"  [{bar}] {dbfs:6.1f} dBFS  listening: {name[:34]}"


# ---------------------------------------------------------------------------
# Segmenter + inference (the consumer loop)
# ---------------------------------------------------------------------------
def run_transcription(model, source):
    rate = source.rate
    status = StatusLine()

    utterance = []          # native-rate float32 chunks in [-1, 1]
    utt_sec = 0.0           # total buffered audio (speech + internal/trailing silence)
    speech_sec = 0.0        # just the speech blocks — what the MIN_UTT_SEC blip filter gates on
    speaking = False
    silence_sec = 0.0

    def flush():
        nonlocal utterance, utt_sec, speech_sec, speaking, silence_sec
        if speech_sec >= MIN_UTT_SEC:
            native = np.concatenate(utterance).astype(np.float32)
            audio = soxr.resample(native, rate, WHISPER_SR).astype(np.float32)
            t_infer = time.time()
            segments, _ = model.transcribe(
                audio, language=LANGUAGE, beam_size=BEAM_SIZE, vad_filter=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                dur = len(audio) / WHISPER_SR
                lag = time.time() - t_infer
                stamp = time.strftime("%H:%M:%S")
                status.message(f"[{stamp}] {text}    ({dur:.1f}s audio, {lag:.2f}s to transcribe)")
        utterance = []
        utt_sec = 0.0
        speech_sec = 0.0
        speaking = False
        silence_sec = 0.0

    now = time.time()
    last_block_t = now      # last time ANY block arrived
    last_sound_t = now      # last time a block carried real sound
    warned_idle = False
    warned_silent = False
    clean_eof = False
    reported_drops = 0

    try:
        while True:
            try:
                raw = audio_q.get(timeout=0.25)
                if raw is None:                 # end-of-stream / error sentinel
                    clean_eof = not capture_error
                    break
            except queue.Empty:
                raw = None                      # no data this tick — fall to watchdog

            now = time.time()

            if raw is not None:
                block16 = np.frombuffer(raw, dtype=np.int16)
                if block16.size:
                    last_block_t = now
                    warned_idle = False
                    rms = float(np.sqrt(np.mean(block16.astype(np.float32) ** 2)))
                    if rms > SILENCE_RMS_FLOOR:
                        last_sound_t = now
                        warned_silent = False
                    if source.is_live:
                        status.status(_vu_meter(rms, source.name))

                    block_f = block16.astype(np.float32) / 32768.0
                    block_sec = len(block_f) / rate
                    if rms > SILENCE_RMS:
                        speaking = True
                        silence_sec = 0.0
                        utterance.append(block_f)
                        utt_sec += block_sec
                        speech_sec += block_sec
                    elif speaking:
                        utterance.append(block_f)
                        utt_sec += block_sec
                        silence_sec += block_sec
                        if silence_sec >= END_SILENCE:
                            flush()
                    if utt_sec >= MAX_UTT_SEC:
                        flush()

            # Two-signal audio-health watchdog (live sources only).
            if source.is_live:
                if now - last_block_t > NO_BLOCKS_WARN_SEC:
                    if not warned_idle:
                        status.message(
                            f"(no audio from '{source.name}' in "
                            f"{NO_BLOCKS_WARN_SEC:.0f}s - idle/phantom endpoint? play some "
                            f"audio, or pick another with --list-devices / --device)")
                        warned_idle = True
                elif now - last_sound_t > SILENCE_WARN_SEC and not warned_silent:
                    status.message(
                        f"(audio from '{source.name}' but near-silent for "
                        f"{SILENCE_WARN_SEC:.0f}s - wrong or muted output device?)")
                    warned_silent = True

            if dropped_blocks - reported_drops >= 50:   # ~5 s of audio dropped
                reported_drops = dropped_blocks
                status.message(f"([behind] dropped {dropped_blocks} audio blocks - "
                               f"inference slower than realtime)")

        if clean_eof:
            flush()          # transcribe whatever was still buffered at a clean end
        stop_flag.set()
        if capture_error:
            status.message(f"Audio capture failed: {capture_error[0]}")

    except KeyboardInterrupt:
        status.message("Stopping...")
        stop_flag.set()
    except Exception as e:
        # e.g. a mid-stream transcribe/resample failure — report it, don't crash
        status.message(f"Transcription stopped on error: {e}")
        stop_flag.set()
    finally:
        try:
            source.stop()   # always release the audio device (error / repeat Ctrl+C safe)
        except BaseException:
            pass


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="M0.1 live captions from Windows system audio.")
    ap.add_argument("--list-devices", action="store_true",
                    help="list loopback capture devices and exit")
    ap.add_argument("--device", metavar="SUBSTR",
                    help="use the loopback device whose name contains SUBSTR (remembered)")
    ap.add_argument("--loopback-index", type=int, default=LOOPBACK_INDEX, metavar="N",
                    help="use this exact WASAPI loopback device index")
    ap.add_argument("--pick", action="store_true",
                    help="choose a loopback device interactively (remembered)")
    ap.add_argument("--wav", metavar="PATH",
                    help="transcribe a WAV file instead of live capture (deterministic)")
    ap.add_argument("--wav-fast", action="store_true",
                    help="with --wav, feed as fast as possible (no real-time pacing)")
    ap.add_argument("--blocking", action="store_true",
                    help="use the blocking-read capture fallback instead of callback mode")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        p = pyaudio.PyAudio()
        try:
            print_device_list(p)
        finally:
            p.terminate()
        return

    # Build the source FIRST so a bad --wav path or device selection fails fast,
    # before the multi-second model load.
    if args.wav:
        source = WavFileAudioSource(args.wav, paced=not args.wav_fast)
    else:
        p = pyaudio.PyAudio()
        try:
            dev = resolve_loopback(p, args)
        finally:
            p.terminate()
        cls = BlockingLoopbackCapture if args.blocking else LoopbackCapture
        source = cls(dev)

    bootstrap_cuda_dlls()
    model = load_model()

    if source.is_live:
        print(f"Capturing: {source.name}  "
              f"({source.rate} Hz, {source.channels} ch, index {source.index})")
        print("Play some audio (a call, a video...) and speak. Ctrl+C to stop.\n")
    else:
        print(f"Source: {source.name}  ({source.rate} Hz)")
        print("Replaying WAV...\n")

    source.start()
    run_transcription(model, source)
    print("Done.")


if __name__ == "__main__":
    main()
