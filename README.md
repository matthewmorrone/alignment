# Lyric Alignment

`stable-ts`-based tool that turns audio + lyrics into timed `.srt` subtitles and `.TextGrid` files.

Everything lives in `align.py`. Run it anywhere — it auto-installs its own dependencies into `.venv` on first use.

## Quick start

```bash
# CLI: align all wav/txt pairs in the current directory
python3 align.py

# HTTP API
python3 align.py serve
```

That's it. On first run it creates `.venv`, installs Python packages, and (on first request) downloads the Whisper model into `.cache/stable-ts`.

## System requirements

Python 3.10+ and ffmpeg must be present. On Debian/Raspberry Pi OS:

```bash
sudo apt update && sudo apt install -y python3 python3-venv ffmpeg
```

## CLI usage

Place `.wav` files alongside `.txt` files with matching names in any directory, then run:

```bash
cd /path/to/songs
python3 /path/to/align.py
```

Outputs a `.srt` and `.TextGrid` next to each `.wav`.

## HTTP API

```bash
python3 align.py serve                        # default: 0.0.0.0:8000
python3 align.py serve --port 9000
python3 align.py serve --host 127.0.0.1
```

### Health check

```bash
curl http://localhost:8000/health
```

### Align — SRT response (default)

```bash
curl -X POST "http://localhost:8000/align" \
  -F "audio=@song.wav" \
  -F "lyrics=<song.txt" \
  -F "language=ja" \
  -o aligned.srt
```

### Align — JSON response

Pass `format=json` to get SRT, TextGrid, and metadata in one response:

```bash
curl -X POST "http://localhost:8000/align" \
  -F "audio=@song.wav" \
  -F "lyrics=<song.txt" \
  -F "language=ja" \
  -F "format=json"
```

Response:

```json
{
  "srt": "...",
  "textgrid": "...",
  "segment_count": 42,
  "model": "small",
  "language": "ja"
}
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `STABLE_TS_MODEL` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `STABLE_TS_DEVICE` | auto | `cpu`, `cuda`, etc. |
| `STABLE_TS_LANGUAGE` | `ja` | Default language code |
| `STABLE_TS_DOWNLOAD_ROOT` | `.cache/stable-ts` | Where the model is cached |

On a Raspberry Pi, `STABLE_TS_MODEL=base` is a good starting point.

## Raspberry Pi / systemd

Copy the project folder to the Pi:

```bash
rsync -av /path/to/alignment pi@<pi-ip>:/home/pi/alignment
```

Edit `lyric-align.service` if needed, then install:

```bash
sudo cp lyric-align.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lyric-align
sudo systemctl start lyric-align
sudo systemctl status lyric-align
```

The included unit runs:

```
/usr/bin/python3 /home/pi/alignment/align.py serve --host 0.0.0.0 --port 8000
```

## Diagnostics

```bash
python3 align.py doctor           # print environment status
python3 align.py doctor --strict  # exit non-zero if anything is wrong
python3 align.py install          # (re-)install dependencies manually
```

## Troubleshooting

- `ffmpeg=missing` → `sudo apt install -y ffmpeg`
- Python too old → install Python 3.10+
- Package install failed → run `python3 align.py install`
- Slow on Pi → set `STABLE_TS_MODEL=base`, keep `workers=1`
- First request is always slower (model download + load)
