"""Settings — pydantic-settings with a TOML file source, plus device persistence.

Precedence (highest first): explicit kwargs (CLI) > LC_* env vars > the TOML file
> field defaults. The TOML file lives in the platform config dir
(``%APPDATA%\\live-captions\\config.toml`` on Windows).
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional, Tuple, Type

import platformdirs
import tomli_w
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

APP_NAME = "live-captions"
CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME, appauthor=False))
CONFIG_PATH = CONFIG_DIR / "config.toml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LC_", extra="ignore")

    # model / compute
    model_name: str = "medium"
    device: str = "auto"            # auto / cuda / cpu
    gpu_compute: str = "float16"
    cpu_compute: str = "int8"
    language: Optional[str] = "en"
    beam_size: int = 1

    # segmenter
    block_sec: float = 0.1
    silence_rms: float = 350.0
    end_silence_sec: float = 0.6
    min_utt_sec: float = 0.4
    max_utt_sec: float = 12.0

    # device selection (persisted by save_device_choice)
    loopback_name: Optional[str] = None
    loopback_ordinal: int = 0

    # audio-health watchdog
    silence_rms_floor: float = 5.0
    no_blocks_warn_sec: float = 4.0
    silence_warn_sec: float = 8.0

    # diarization (offline "who is talking" pass)
    diarizer: str = "auto"                 # auto | pyannote | sherpa
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    diarize_threshold: float = 0.5         # sherpa clustering threshold (higher = fewer speakers)
    diarize_num_speakers: int = -1         # -1 = infer
    #: optional override; normally the token comes from `huggingface-cli login`
    #: or the HF_TOKEN env var — prefer those over putting a secret in config.toml
    hf_token: Optional[str] = None

    # live diarization (Streaming Sortformer). CPU by default: ~RTF 0.4, and it
    # keeps the GPU's VRAM entirely for the Whisper worker.
    diarize_live_device: str = "cpu"
    diarize_live_threshold: float = 0.5

    # streaming ASR (M3)
    stream_process_interval: float = 0.5   # seconds of new audio between decode passes
    stream_end_silence_sec: float = 0.8    # trailing silence (VAD) that finalizes a line
    stream_max_line_sec: float = 8.0       # force-finalize a line this long even without a pause
    stream_max_buffer_sec: float = 15.0    # rolling buffer cap
    stream_vad_threshold: float = 0.5

    # overlay (M2)
    overlay_font_pt: int = 24
    overlay_max_lines: int = 3
    overlay_opacity: float = 1.0
    overlay_width_frac: float = 0.7   # max pill width as a fraction of screen width
    overlay_text_color: str = "#FFFFFF"   # base caption colour (speakers override when coloured)
    speaker_colors: bool = False          # colour captions by speaker (turns on live diarization)
    open_settings_on_launch: bool = True  # show the Settings window when the app starts
    overlay_movable: bool = False         # draggable (vs click-through); remembered across launches
    # What happens to captions when the app opens. "resume" honours how you left it,
    # which is why it is the default: a fixed checkbox cannot express "I pressed Start
    # last time, so start". "always" / "never" pin it either way.
    startup_mode: str = "resume"          # resume | always | never
    last_transport_state: str = "running"  # updated as you Start/Pause/Stop
    start_captions_on_launch: bool = True  # legacy; migrated into startup_mode
    settings_tab: int = 0                 # reopen the control panel on the tab you left on

    # Windows applies the output volume before we capture, so quiet playback would
    # otherwise stop triggering the VAD. This scales it back up (mute is still mute).
    auto_gain: bool = True
    auto_gain_target_rms: float = 0.05
    auto_gain_max: float = 30.0

    # global hotkeys (Win32 RegisterHotKey). Remap here if another app claims one.
    hotkey_toggle: str = "ctrl+alt+c"     # show/hide the overlay
    hotkey_pause: str = "ctrl+alt+p"      # pause/resume captions
    hotkey_quit: str = "ctrl+alt+q"       # quit the app
    hotkey_left: str = "ctrl+alt+left"
    hotkey_right: str = "ctrl+alt+right"
    hotkey_up: str = "ctrl+alt+up"
    hotkey_down: str = "ctrl+alt+down"
    hotkey_nudge_px: int = 40
    hotkeys_enabled: bool = True

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        sources = [init_settings, env_settings]
        if CONFIG_PATH.exists():
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=str(CONFIG_PATH)))
        return tuple(sources)


def save_settings(**kwargs) -> None:
    """Merge the given keys into the TOML config, preserving other keys. A value
    of None removes the key (reverting to the field default) — TOML has no null."""
    data = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
    if not isinstance(data, dict):
        data = {}
    for key, value in kwargs.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "wb") as f:
        tomli_w.dump(data, f)


def save_device_choice(name: str, ordinal: int) -> None:
    """Persist the chosen loopback (by name + ordinal)."""
    save_settings(loopback_name=name, loopback_ordinal=ordinal)
