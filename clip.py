import json
import os
import subprocess
from pathlib import Path

import requests
from faster_whisper import WhisperModel

URL = os.environ["VIDEO_URL"]
NUM = int(os.environ.get("NUM_CLIPS", "5"))
WHISPER = os.environ.get("WHISPER_MODEL", "base")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
KEY = os.environ["GEMINI_API_KEY"]

WORK = Path("work")
OUT = Path("clips")
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


def download():
    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "-o", str(WORK / "source.%(ext)s"),
        URL,
    ]
    cookies = os.environ.get("YT_COOKIES", "").strip()
    if cookies:
        Path("cookies.txt").write_text(cookies)
        cmd[1:1] = ["--cookies", "cookies.txt"]
    subprocess.run(cmd, check=True)
    return WORK / "source.mp4"


def transcribe(path):
    model = WhisperModel(WHISPER, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), word_timestamps=True, vad_filter=True)
    out = []
    for s in segments:
        out.append({
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
            "words": [
                {"s": w.start, "e": w.end, "w": w.word.strip()}
                for w in (s.words or [])
            ],
        })
    return out


def pick_moments(segs):
    transcript = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segs
    )[:400000]
    prompt = f"""You are an expert short-form video editor for TikTok, Reels and Shorts.
Below is a timestamped transcript. Pick the {NUM} most viral moments: strong hooks,
surprising or funny lines, emotional peaks, hot takes, great stories.

Rules:
- Each clip must be 20 to 60 seconds long.
- Start on a line that grabs attention and end on a complete thought.
- Use start/end times (in seconds) taken from the transcript timestamps.
- Clips must not overlap.
- Return ONLY a JSON array of objects: {{"start": number, "end": number, "title": string, "score": number}}
  where score is 1-100 virality.

Transcript:
{transcript}
"""
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=180,
    )
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def snap(segs, start, end):
    s = min(segs, key=lambda x: abs(x["start"] - start))["start"]
    e = min(segs, key=lambda x: abs(x["end"] - end))["end"]
    if e <= s:
        e = s + 30
    return s, min(e, s + 75)


def ts(t):
    ms = int(round(max(t, 0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"


def make_srt(segs, a, b, path):
    words = [
        w for s in segs for w in s["words"]
        if w["s"] >= a - 0.05 and w["e"] <= b + 0.05
    ]
    blocks = []
    for i in range(0, len(words), 3):
        g = words[i:i + 3]
        st = g[0]["s"] - a
        en = max(g[-1]["e"] - a, st + 0.1)
        text = " ".join(x["w"] for x in g).upper()
        blocks.append(f"{len(blocks) + 1}\n{ts(st)} --> {ts(en)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def cut(src, a, b, name):
    make_srt(SEGS, a, b, OUT / f"{name}.srt")
    style = (
        "FontName=DejaVu Sans,Bold=1,FontSize=10,Outline=2,"
        "PrimaryColour=&H00FFFFFF,Alignment=2,MarginV=45"
    )
    vf = (
        "crop='min(iw,ih*9/16)':ih,scale=1080:1920,"
        f"subtitles={name}.srt:force_style='{style}'"
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{a:.2f}", "-i", str(src.resolve()),
        "-t", f"{b - a:.2f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        f"{name}.mp4",
    ], cwd=OUT, check=True)


if __name__ == "__main__":
    src = download()
    SEGS = transcribe(src)
    if not SEGS:
        raise SystemExit("No speech found in this video.")
    moments = pick_moments(SEGS)
    summary = []
    for i, m in enumerate(moments[:NUM], 1):
        a, b = snap(SEGS, float(m["start"]), float(m["end"]))
        name = f"clip{i}"
        cut(src, a, b, name)
        summary.append(f"{name}.mp4 | {m.get('title', '')} | score {m.get('score', '')}")
    (OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
