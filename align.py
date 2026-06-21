import argparse
import importlib.util
import json as json_module
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unicodedata
import venv
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    ".align_runtime",
    "vendor",
}

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
MIN_PYTHON = (3, 10)
REQUIRED_MODULES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "stable_whisper": "stable-ts",
    "praatio": "praatio",
    "multipart": "python-multipart",
}

DEFAULT_MODEL = os.environ.get("STABLE_TS_MODEL", "large-v3")
DEFAULT_DEVICE = os.environ.get("STABLE_TS_DEVICE")
DEFAULT_LANGUAGE = os.environ.get("STABLE_TS_LANGUAGE", "ja")
DOWNLOAD_ROOT = os.environ.get(
    "STABLE_TS_DOWNLOAD_ROOT",
    str(PROJECT_ROOT / ".cache" / "stable-ts"),
)

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

DEFAULT_VAD = _env_bool("STABLE_TS_VAD", True)
DEFAULT_DENOISER = os.environ.get("STABLE_TS_DENOISER") or None
DEFAULT_KEEP_ALL_LINES = _env_bool("STABLE_TS_KEEP_ALL_LINES", True)


def _env_float_loose(name: str, default: float) -> float:
    """Like _env_float but defined early so VAD threshold can use it."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Silero VAD threshold (0.0 = everything is speech, 1.0 = nothing is).
# Default 0.35 is stable-ts's own default. Lower for music where vocals get
# mis-classified as non-speech (e.g., heavily processed/layered vocals); raise
# to be more aggressive about cutting silence. Only applied when vad=True.
DEFAULT_VAD_THRESHOLD = _env_float_loose("STABLE_TS_VAD_THRESHOLD", 0.35)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Music interval marking — insert synthetic segments labeled MUSIC_LABEL wherever
# there's a gap >= MUSIC_MIN_GAP seconds between aligned lyric lines (or before
# the first / after the last). Set MUSIC_LABEL to "" to disable.
DEFAULT_MUSIC_LABEL = os.environ.get("STABLE_TS_MUSIC_LABEL", "♪")
DEFAULT_MUSIC_MIN_GAP = _env_float("STABLE_TS_MUSIC_MIN_GAP", 3.0)
# Stable-ts often stretches `segment.end` toward the next segment's start, which
# swallows instrumental breaks into the previous lyric. We instead use the last
# word's end time as the segment's effective end, plus a small buffer to allow
# for sustained vowels (e.g., the final ン in シェノン held past Whisper's word boundary).
DEFAULT_MUSIC_TAIL_BUFFER = _env_float("STABLE_TS_MUSIC_TAIL_BUFFER", 0.5)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    wav_path: Path
    txt_path: Path | None


@dataclass
class AlignmentArtifacts:
    srt_path: Path
    textgrid_path: Path
    json_path: Path
    segment_count: int


# ---------------------------------------------------------------------------
# Core alignment logic
# ---------------------------------------------------------------------------

_MODEL_LOCK = threading.Lock()
_CACHED_MODEL = None
_CACHED_MODEL_KEY: tuple[str, str | None, str | None] | None = None


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def pretty_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def should_skip_dir(name: str) -> bool:
    return name.startswith(".") or name in SKIP_DIR_NAMES


def is_alignment_up_to_date(pair: Pair) -> bool:
    """Return True if the .srt for this pair already exists and is newer than
    both the audio file (.mp3/.wav/etc.) and the lyrics file (.txt).

    When this returns True, discover_pairs will filter the pair out (unless
    --force is set), so re-running on a directory doesn't redo work.

    Notes for the implementer:
      - The expected output path is `pair.wav_path.with_suffix(".srt")`.
      - If pair.txt_path is None (transcribe mode), only compare against the audio.
      - Use `Path.exists()` and `Path.stat().st_mtime`.
      - Be conservative: if anything looks off (file missing, error reading stat),
        return False so we re-process rather than silently skipping.
    """
    srt = pair.wav_path.with_suffix(".srt")
    try:
        if not srt.exists():
            return False
        srt_mtime = srt.stat().st_mtime
        if srt_mtime < pair.wav_path.stat().st_mtime:
            return False
        if pair.txt_path is not None and srt_mtime < pair.txt_path.stat().st_mtime:
            return False
        return True
    except OSError:
        return False


def discover_pairs(root: Path, *, force: bool = False) -> tuple[list[Pair], int]:
    """Discover (audio, txt) pairs under root.

    Returns (pairs_to_process, skipped_count). Pairs whose outputs are already
    up to date are filtered out unless force=True.
    """
    all_pairs: list[Pair] = []

    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not should_skip_dir(name)]
        current_dir = Path(dirpath)

        txt_by_stem = {
            normalize_text(path.stem): path for path in current_dir.glob("*.txt")
        }

        audio_files = [
            p for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg")
            for p in current_dir.glob(ext)
        ]
        for wav_path in sorted(audio_files):
            txt_path = txt_by_stem.get(normalize_text(wav_path.stem))
            all_pairs.append(Pair(wav_path=wav_path, txt_path=txt_path))

    all_pairs.sort(key=lambda pair: pretty_path(pair.wav_path))

    if force:
        return all_pairs, 0

    fresh = [p for p in all_pairs if not is_alignment_up_to_date(p)]
    return fresh, len(all_pairs) - len(fresh)


def normalize_lyrics_text(text: str) -> str:
    lines = [
        normalize_text(line.strip())
        for line in text.splitlines()
        if line.strip()
    ]
    return "\n".join(lines)


def read_lyrics(path: Path) -> str:
    return normalize_lyrics_text(path.read_text(encoding="utf-8"))


def get_model(
    model_name: str = DEFAULT_MODEL,
    device: str | None = DEFAULT_DEVICE,
    download_root: str | None = DOWNLOAD_ROOT,
):
    global _CACHED_MODEL, _CACHED_MODEL_KEY

    key = (model_name, device, download_root)
    with _MODEL_LOCK:
        if _CACHED_MODEL is None or _CACHED_MODEL_KEY != key:
            if download_root:
                Path(download_root).mkdir(parents=True, exist_ok=True)
            import stable_whisper
            _CACHED_MODEL = stable_whisper.load_model(
                model_name,
                device=device,
                download_root=download_root,
            )
            _CACHED_MODEL_KEY = key
        return _CACHED_MODEL


def run_alignment(
    audio_path: Path,
    lyrics_text: str | None = None,
    *,
    language: str = DEFAULT_LANGUAGE,
    model_name: str = DEFAULT_MODEL,
    device: str | None = DEFAULT_DEVICE,
    download_root: str | None = DOWNLOAD_ROOT,
    vad: bool = DEFAULT_VAD,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    denoiser: str | None = DEFAULT_DENOISER,
    keep_all_lines: bool = DEFAULT_KEEP_ALL_LINES,
):
    model = get_model(model_name=model_name, device=device, download_root=download_root)

    if lyrics_text:
        normalized_lyrics = normalize_lyrics_text(lyrics_text)
        if not normalized_lyrics:
            raise RuntimeError("Transcript is empty after removing blank lines.")
        result = model.align(
            str(audio_path),
            normalized_lyrics,
            language=language,
            original_split=True,
            regroup=False,
            vad=vad,
            vad_threshold=vad_threshold,
            denoiser=denoiser,
            verbose=False,
        )
    else:
        result = model.transcribe(
            str(audio_path),
            language=language,
            vad=vad,
            vad_threshold=vad_threshold,
            denoiser=denoiser,
            verbose=False,
        )

    if result is None:
        raise RuntimeError("stable-ts returned no result.")

    if not keep_all_lines:
        result.remove_no_word_segments()
    return result


def get_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds via ffprobe. Returns 0.0 on failure."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            text=True,
        )
        return float(out.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return 0.0


def insert_music_intervals(
    segments: list[tuple[float, float, str]],
    duration: float,
    min_gap: float,
    label: str,
) -> list[tuple[float, float, str]]:
    """Return segments augmented with synthetic `label` intervals wherever
    the gap between consecutive lyric lines (or before first / after last)
    is >= min_gap seconds. Original segments are preserved unchanged."""
    if not label or min_gap <= 0:
        return list(segments)
    sorted_segs = sorted(segments, key=lambda s: s[0])
    augmented: list[tuple[float, float, str]] = []
    # Leading instrumental
    if sorted_segs:
        first_start = sorted_segs[0][0]
        if first_start >= min_gap:
            augmented.append((0.0, first_start, label))
    elif duration >= min_gap:
        return [(0.0, duration, label)]
    # Interleaved
    for i, seg in enumerate(sorted_segs):
        augmented.append(seg)
        if i + 1 < len(sorted_segs):
            gap = sorted_segs[i + 1][0] - seg[1]
            if gap >= min_gap:
                augmented.append((seg[1], sorted_segs[i + 1][0], label))
    # Trailing instrumental
    if sorted_segs and duration > 0:
        last_end = sorted_segs[-1][1]
        if duration - last_end >= min_gap:
            augmented.append((last_end, duration, label))
    return augmented


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[tuple[float, float, str]], output_path: Path) -> None:
    lines: list[str] = []
    for i, (start, end, text) in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_textgrid(
    segments: list[tuple[float, float, str]],
    word_intervals: list[tuple[float, float, str]],
    duration: float,
    output_path: Path,
) -> None:
    from praatio import textgrid

    seg_entries = [(s, e, t) for s, e, t in segments if e > s]
    word_entries = [(s, e, t) for s, e, t in word_intervals if e > s]

    max_time = max(
        duration,
        max((e for _, e, _ in seg_entries), default=0.0),
        max((e for _, e, _ in word_entries), default=0.0),
    )

    tg = textgrid.Textgrid()
    tg.addTier(textgrid.IntervalTier("segments", seg_entries, minT=0.0, maxT=max_time))
    tg.addTier(textgrid.IntervalTier("words", word_entries, minT=0.0, maxT=max_time))
    tg.save(
        str(output_path),
        format="short_textgrid",
        includeBlankSpaces=True,
    )


def save_alignment_artifacts(
    result,
    output_stem: Path,
    audio_path: Path,
    *,
    lyrics_text: str | None = None,
    music_label: str = DEFAULT_MUSIC_LABEL,
    music_min_gap: float = DEFAULT_MUSIC_MIN_GAP,
) -> AlignmentArtifacts:
    srt_path = output_stem.with_suffix(".srt")
    textgrid_path = output_stem.with_suffix(".TextGrid")
    json_path = output_stem.with_suffix(".json")

    # Write the canonical stable-ts JSON first, before any of our SRT/TextGrid
    # post-processing. This preserves the full WhisperResult (segments + per-word
    # timestamps + probabilities + everything else) verbatim on disk, so derived
    # formats can always be regenerated without re-running the model. If a future
    # bug in our SRT/TextGrid writers loses information, the JSON is the source
    # of truth and the fix is a re-conversion, not a re-alignment.
    result.save_as_json(str(json_path))

    # Extract non-empty segments preserving stable-ts's order. We don't filter on
    # duration here — even zero-duration (failed alignment) segments stay in the
    # list so KEEP_ALL_LINES semantics are honored.
    MIN_PLACEHOLDER_DURATION = 0.1
    FAIL_THRESHOLD = 0.05  # below this = treat as a failed alignment
    MIN_MEANINGFUL_SILENCE = 0.3  # silences shorter than this are inter-syllable, not music

    # Preserve original TXT lines (forced-alignment contract). stable-ts can return
    # mangled text when alignment fails (character-shifted, hallucinated, etc.)
    # because of how `original_split=True` re-tokenizes by char-count rather than
    # by line break. Since stable-ts returns segments in input-line order, we just
    # override each segment's text with the corresponding input line by index.
    input_lines: list[str] | None = None
    if lyrics_text:
        normalized = normalize_lyrics_text(lyrics_text)
        input_lines = [ln for ln in normalized.split("\n") if ln]

    # Silero VAD's silence intervals (already computed by stable-ts during alignment).
    # We use these to tighten segment ends that stretched past actual singing.
    raw_ns = getattr(result, "nonspeech_sections", None) or []
    nonspeech: list[tuple[float, float]] = []
    for ns in raw_ns:
        if isinstance(ns, dict):
            nonspeech.append((float(ns["start"]), float(ns["end"])))
        else:
            nonspeech.append((float(ns[0]), float(ns[1])))

    raw_segments: list[tuple[float, float, str]] = []
    word_intervals: list[tuple[float, float, str]] = []
    for seg_idx, segment in enumerate(result.segments):
        start = float(segment.start)
        end = float(segment.end)

        # Use the original TXT line if we have it. This guarantees the SRT text
        # matches your input regardless of whether stable-ts produced corrupted
        # tokens. Fall back to stable-ts's segment.text only if no lyrics were
        # provided (transcription mode rather than alignment mode).
        if input_lines is not None and seg_idx < len(input_lines):
            text_value = input_lines[seg_idx]
        else:
            text_value = str(segment.text).strip()
        if not text_value:
            continue

        # Gather word-level intervals AND track the last aligned word's end time.
        last_word_end = 0.0
        for word in segment.words or []:
            word_start = float(word.start)
            word_end = float(word.end)
            word_text = str(word.word).strip()
            if word_end > word_start and word_text:
                word_intervals.append((word_start, word_end, word_text))
                if word_end > last_word_end:
                    last_word_end = word_end

        # Tighten the segment's end using two signals:
        #   1. last word's end + small buffer (handles Whisper-honest alignments
        #      where segment.end stretched past the last word naturally)
        #   2. first meaningful silence (>= MIN_MEANINGFUL_SILENCE) that begins
        #      at or after the last word's end but before segment.end — this is
        #      audio-grounded evidence of where the singing actually stopped
        # We take the minimum (earliest) of these candidates, never going below
        # segment.start. If neither signal is available, segment.end stays.
        if last_word_end > 0.0:
            tightened = min(end, last_word_end + DEFAULT_MUSIC_TAIL_BUFFER)
            for ns_start, ns_end in nonspeech:
                if (ns_end - ns_start) < MIN_MEANINGFUL_SILENCE:
                    continue
                if last_word_end <= ns_start < end:
                    tightened = min(tightened, ns_start)
                    break
            end = max(start, tightened)
        raw_segments.append((start, end, text_value))

    raw_segments.sort(key=lambda s: s[0])

    # Redistribute failed alignments: when a run of failed (near-zero-duration)
    # segments is followed by an aligned segment starting at the same time, split
    # that segment's time slot among the failed lines + itself, proportional to
    # text character length. This unwinds stable-ts's tendency to dump the audio
    # for an unaligned line into the next line, leaving the unaligned one at 1ms.
    redistributed: list[tuple[float, float, str]] = []
    i = 0
    while i < len(raw_segments):
        s, e, t = raw_segments[i]
        if e - s < FAIL_THRESHOLD:
            run = [i]
            j = i + 1
            while j < len(raw_segments) and raw_segments[j][1] - raw_segments[j][0] < FAIL_THRESHOLD:
                run.append(j)
                j += 1
            if j < len(raw_segments):
                next_s, next_e, next_t = raw_segments[j]
                run_start = raw_segments[run[0]][0]
                # Only redistribute if the next segment starts at-or-near the run
                if abs(next_s - run_start) < 0.5:
                    failed_texts = [raw_segments[k][2] for k in run]
                    all_texts = failed_texts + [next_t]
                    char_lens = [max(len(t), 1) for t in all_texts]
                    total_chars = sum(char_lens)
                    pool_duration = next_e - next_s
                    cursor = next_s
                    for idx, length in enumerate(char_lens[:-1]):
                        share = (length / total_chars) * pool_duration
                        redistributed.append((cursor, cursor + share, all_texts[idx]))
                        cursor += share
                    redistributed.append((cursor, next_e, next_t))
                    i = j + 1
                    continue
        redistributed.append((s, e, t))
        i += 1
    raw_segments = redistributed

    # Clamp overlaps so consecutive segments don't intersect (praatio's IntervalTier
    # rejects overlapping intervals). Any failed segment that didn't redistribute
    # gets a 1ms epsilon as a last resort.
    clamped: list[tuple[float, float, str]] = []
    prev_end = 0.0
    for i, (s, e, t) in enumerate(raw_segments):
        next_start = raw_segments[i + 1][0] if i + 1 < len(raw_segments) else float("inf")
        s = max(s, prev_end)
        e = min(e, next_start)
        if e <= s:
            e = s + 0.001
        clamped.append((s, e, t))
        prev_end = e
    raw_segments = clamped

    duration = get_audio_duration(audio_path)
    if duration <= 0:
        duration = max((e for _, e, _ in raw_segments), default=0.0)

    augmented = insert_music_intervals(raw_segments, duration, music_min_gap, music_label)

    write_srt(augmented, srt_path)
    build_textgrid(augmented, word_intervals, duration, textgrid_path)

    return AlignmentArtifacts(
        srt_path=srt_path,
        textgrid_path=textgrid_path,
        json_path=json_path,
        segment_count=len(raw_segments),
    )


def align_audio_to_outputs(
    audio_path: Path,
    lyrics_text: str | None,
    output_stem: Path,
    *,
    language: str = DEFAULT_LANGUAGE,
    model_name: str = DEFAULT_MODEL,
    device: str | None = DEFAULT_DEVICE,
    download_root: str | None = DOWNLOAD_ROOT,
) -> AlignmentArtifacts:
    result = run_alignment(
        audio_path,
        lyrics_text,
        language=language,
        model_name=model_name,
        device=device,
        download_root=download_root,
    )
    return save_alignment_artifacts(result, output_stem, audio_path, lyrics_text=lyrics_text)


# ---------------------------------------------------------------------------
# HTTP service
# ---------------------------------------------------------------------------

def make_app():
    from typing import Annotated
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse

    app = FastAPI(title="Lyric Alignment API", version="1.0.0")

    async def save_upload(upload: UploadFile, destination: Path) -> None:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    @app.get("/health")
    def health() -> dict[str, str | None]:
        return {
            "status": "ok",
            "model": DEFAULT_MODEL,
            "language": DEFAULT_LANGUAGE,
            "device": DEFAULT_DEVICE,
            "download_root": DOWNLOAD_ROOT,
            "ffmpeg": shutil.which("ffmpeg"),
        }

    @app.post("/align")
    async def align(
        audio: Annotated[UploadFile, File()],
        lyrics: Annotated[str | None, Form()] = None,
        language: Annotated[str, Form()] = DEFAULT_LANGUAGE,
        model: Annotated[str, Form()] = DEFAULT_MODEL,
        format: Annotated[str, Form()] = "srt",
    ):
        suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
        try:
            with tempfile.TemporaryDirectory(prefix="lyric_align_") as tmpdir:
                tmpdir_path = Path(tmpdir)
                audio_path = tmpdir_path / f"input{suffix}"
                output_stem = tmpdir_path / "aligned"
                await save_upload(audio, audio_path)
                artifacts = align_audio_to_outputs(
                    audio_path, lyrics, output_stem,
                    language=language, model_name=model,
                    device=DEFAULT_DEVICE, download_root=DOWNLOAD_ROOT,
                )
                srt_text = artifacts.srt_path.read_text(encoding="utf-8")
                tg_text = artifacts.textgrid_path.read_text(encoding="utf-8")
                # stable-ts JSON is the full WhisperResult, parsed into a dict
                # so it nests naturally inside the response envelope rather than
                # being a stringified blob.
                stable_ts_json = json_module.loads(artifacts.json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if format == "json":
            return JSONResponse({
                "srt": srt_text,
                "textgrid": tg_text,
                "stable_ts": stable_ts_json,
                "segment_count": artifacts.segment_count,
                "model": model,
                "language": language,
            })

        return PlainTextResponse(
            srt_text,
            media_type="application/x-subrip",
            headers={"Content-Disposition": 'attachment; filename="aligned.srt"'},
        )

    return app


# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

def venv_python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def current_python_too_old() -> bool:
    return sys.version_info < MIN_PYTHON


def current_python_label() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def current_missing_modules() -> list[str]:
    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def python_has_required_modules(python_executable: Path) -> bool:
    probe = """
import importlib.util
import sys
modules = ['fastapi', 'uvicorn', 'stable_whisper', 'praatio', 'multipart']
missing = [name for name in modules if importlib.util.find_spec(name) is None]
sys.exit(0 if not missing else 1)
""".strip()
    result = subprocess.run(
        [str(python_executable), "-c", probe],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def require_python_version() -> None:
    if current_python_too_old():
        wanted = ".".join(str(part) for part in MIN_PYTHON)
        raise RuntimeError(f"Python {wanted}+ is required. Found {current_python_label()} at {sys.executable}.")


def require_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    raise RuntimeError("ffmpeg was not found in PATH. Install it first, for example: sudo apt install ffmpeg")


def print_status() -> None:
    ffmpeg_path = shutil.which("ffmpeg") or "missing"
    venv_python = venv_python_path()
    print(f"project_root={PROJECT_ROOT}")
    print(f"python={sys.executable}")
    print(f"python_version={current_python_label()}")
    print(f"ffmpeg={ffmpeg_path}")
    print(f"venv={VENV_DIR}")
    print(f"venv_python_exists={venv_python.exists()}")
    print(f"current_missing_modules={','.join(current_missing_modules()) or 'none'}")
    if venv_python.exists():
        venv_ready = python_has_required_modules(venv_python)
        print("venv_missing_modules=" + ("none" if venv_ready else "missing"))


def run_checked(command: list[str]) -> None:
    print("+", " ".join(command))
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def ensure_venv() -> Path:
    python_executable = venv_python_path()
    if python_executable.exists():
        return python_executable

    print(f"Creating virtual environment in {VENV_DIR}")
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt")
    builder.create(VENV_DIR)

    if not python_executable.exists():
        raise RuntimeError(f"Virtual environment creation failed: {python_executable} not found.")
    return python_executable


def install_dependencies() -> Path:
    require_python_version()
    require_ffmpeg()
    if not REQUIREMENTS_FILE.exists():
        raise RuntimeError(f"Missing requirements file: {REQUIREMENTS_FILE}")

    python_executable = ensure_venv()
    run_checked([str(python_executable), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    run_checked([str(python_executable), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    return python_executable


def reexec_into(python_executable: Path, args: list[str]) -> None:
    os.execv(str(python_executable), [str(python_executable), str(Path(__file__).resolve()), *args])


def ensure_runtime(forwarded_args: list[str]) -> None:
    require_python_version()
    require_ffmpeg()

    missing = current_missing_modules()
    if not missing:
        return

    venv_python = venv_python_path()
    venv_ready = venv_python.exists() and python_has_required_modules(venv_python)
    if not venv_ready:
        install_dependencies()

    reexec_into(venv_python, forwarded_args)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_align(force: bool = False) -> int:
    ensure_runtime(["align"] + (["--force"] if force else []))
    root = Path.cwd()
    pairs, skipped = discover_pairs(root, force=force)

    print(f"Working directory: {root}")
    print(f"Found {len(pairs) + skipped} audio file(s); skipping {skipped} already up to date.")

    if not pairs:
        return 0

    print(
        f"Loading stable-ts model '{DEFAULT_MODEL}'"
        + (f" on '{DEFAULT_DEVICE}'" if DEFAULT_DEVICE else "")
        + f" for language '{DEFAULT_LANGUAGE}'."
    )

    failures = 0
    for index, pair in enumerate(pairs, 1):
        mode = "align" if pair.txt_path else "transcribe"
        print(f"[{index}/{len(pairs)}] {pretty_path(pair.wav_path)} ({mode})")
        try:
            lyrics = read_lyrics(pair.txt_path) if pair.txt_path else None
            artifacts = align_audio_to_outputs(
                pair.wav_path,
                lyrics,
                pair.wav_path.with_suffix(""),
                language=DEFAULT_LANGUAGE,
                model_name=DEFAULT_MODEL,
                device=DEFAULT_DEVICE,
                download_root=DOWNLOAD_ROOT,
            )
            print(
                "  saved: "
                f"{pretty_path(artifacts.srt_path)}, "
                f"{pretty_path(artifacts.textgrid_path)}, "
                f"{pretty_path(artifacts.json_path)} | "
                f"segments={artifacts.segment_count}"
            )
        except Exception as exc:
            failures += 1
            print(f"  [error] {pair.wav_path.stem}: {exc}")

    print(f"Complete. processed={len(pairs) - failures} failed={failures} output_dir={root}")
    return 0 if failures == 0 else 1


def cmd_serve(host: str, port: int, reload_enabled: bool) -> None:
    forwarded_args = ["serve", "--host", host, "--port", str(port)]
    if reload_enabled:
        forwarded_args.append("--reload")

    ensure_runtime(forwarded_args)

    import uvicorn

    uvicorn.run("align:make_app", host=host, port=port, reload=reload_enabled, workers=1, factory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lyric alignment tool and API server.")
    subparsers = parser.add_subparsers(dest="command")

    align_parser = subparsers.add_parser("align", help="Align all audio/txt pairs in the current directory (default).")
    align_parser.add_argument("--force", action="store_true", help="Re-process even when .srt is newer than .mp3 and .txt.")

    install_parser = subparsers.add_parser("install", help="Create .venv and install Python dependencies.")
    install_parser.add_argument("--print-status", action="store_true", help="Print environment status after installation.")

    doctor_parser = subparsers.add_parser("doctor", help="Print environment checks.")
    doctor_parser.add_argument("--strict", action="store_true", help="Exit non-zero when checks fail.")

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP API.")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    raw_args = sys.argv[1:]
    known_cmds = {"align", "install", "doctor", "serve"}
    if not raw_args or raw_args[0] not in known_cmds:
        raw_args = ["align"] + raw_args
    args = parser.parse_args(raw_args)
    command = args.command or "align"

    try:
        if command == "align":
            return cmd_align(force=getattr(args, "force", False))

        if command == "install":
            install_dependencies()
            if args.print_status:
                print_status()
            else:
                print(f"Installed dependencies into {VENV_DIR}")
            return 0

        if command == "doctor":
            missing_modules = current_missing_modules()
            ffmpeg_path = shutil.which("ffmpeg")
            print_status()
            if args.strict and (missing_modules or not ffmpeg_path or current_python_too_old()):
                return 1
            return 0

        if command == "serve":
            cmd_serve(host=args.host, port=args.port, reload_enabled=args.reload)
            return 0

    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
