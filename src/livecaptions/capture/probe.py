"""Find which output device is actually playing sound.

Windows happily presents several render endpoints with byte-identical names — two
monitors on the same GPU produce e.g. two "DELL S2721QS (NVIDIA High Definition
Audio) [Loopback]" entries. Nothing in the name, and no index a user can see,
says which one carries the audio. Worse, a loopback endpoint with nothing playing
through it delivers no data at all rather than silence, so choosing wrong looks
exactly like the app being broken.

So: listen to every loopback at once for a moment and report which one has signal.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from .devices import enumerate_loopbacks

#: RMS above which we call an endpoint "playing". Comfortably above dither/noise
#: and below any real content, including quiet speech.
SIGNAL_RMS = 0.0005


def probe_loopbacks(seconds: float = 4.0) -> List[Tuple[dict, float, int]]:
    """Listen to every loopback endpoint concurrently.

    Returns [(device_info, peak_rms, blocks_seen)] ordered loudest first. Uses ONE
    PyAudio instance and callback streams: a second instance on another thread
    segfaults PortAudio, and a blocking read on an idle endpoint never returns.
    """
    import pyaudiowpatch as pa

    p = pa.PyAudio()
    try:
        devices = enumerate_loopbacks(p)
        stats = {d["index"]: {"rms": 0.0, "blocks": 0} for d in devices}
        streams = []

        def make_cb(idx):
            s = stats[idx]

            def cb(in_data, frame_count, time_info, status):
                d = np.frombuffer(in_data, dtype=np.float32)
                s["blocks"] += 1
                if d.size:
                    s["rms"] = max(s["rms"], float(np.sqrt((d ** 2).mean())))
                return (None, pa.paContinue)
            return cb

        for d in devices:
            try:
                st = p.open(format=pa.paFloat32,
                            channels=min(2, int(d["maxInputChannels"])),
                            rate=int(d["defaultSampleRate"]), input=True,
                            input_device_index=d["index"], frames_per_buffer=1024,
                            stream_callback=make_cb(d["index"]))
                st.start_stream()
                streams.append(st)
            except Exception:
                pass          # an endpoint we can't open simply can't be the answer

        time.sleep(max(0.5, seconds))

        for st in streams:
            try:
                st.stop_stream()
                st.close()
            except Exception:
                pass

        out = [(d, stats[d["index"]]["rms"], stats[d["index"]]["blocks"]) for d in devices]
        out.sort(key=lambda t: t[1], reverse=True)
        return out
    finally:
        p.terminate()


def best_loopback(seconds: float = 4.0):
    """The endpoint currently playing audio, or None if everything was quiet."""
    results = probe_loopbacks(seconds)
    if results and results[0][1] >= SIGNAL_RMS:
        return results[0][0]
    return None
