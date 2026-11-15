import json
import os
import re
import subprocess
import time
from pathlib import Path

import requests
from faster_whisper import WhisperModel

URL = os.environ["VIDEO_URL"]
NUM = int(os.environ.get("NUM_CLIPS", "5"))
WHISPER = os.environ.get("WHISPER_MODEL", "base")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
KEY = os.environ["GEMINI_API_KEY"]

WORK = Path("work")
OUT = Path("clips")
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

MIN_LEN = 15
MAX_LEN = 75


def clean_cookies(raw):
    """Repair a pasted cookies file: strip stray backslashes, restore tabs, dedupe."""
    raw = raw.replace("\\", "")
    lines = ["# Netscape HTTP Cookie File"]
    seen = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line, maxsplit=6)
        if len(parts) == 7:
            row = "\t".join(parts)
            if row not in seen:
                seen.add(row)
                lines.append(row)
    return "\n".join(lines) + "\n"


def download():
    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "-o", str(WORK / "source.%(ext)s"),
        URL,
    ]
    cookies = os.environ.get("YT_COOKIES", "")
    if cookies.strip():
        Path("cookies.txt").write_text(clean_cookies(cookies))
        cmd[1:1] = ["--cookies", "cookies.txt"]
    subprocess.run(cmd, check=True)
    files = sorted(WORK.glob("source.*"))
    mp4 = [f for f in files if f.suffix == ".mp4"]
    return (mp4 or files)[0]


def transcribe(path):
    model = WhisperModel(WHISPER, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), vad_filter=True, word_timestamps=True)
    segs = []
    words = []
    for s in segments:
        text = s.text.strip()
        if text:
            segs.append({"start": s.start, "end": s.end, "text": text})
        for w in s.words or []:
            words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
    return segs, words, info.duration


def ask_gemini(segs, duration):
    transcript = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segs)
    prompt = f"""You are an expert short-form video editor.
Below is a timestamped transcript of a video ({duration:.0f} seconds long).
Pick the {NUM} best moments to turn into standalone vertical clips for TikTok, Reels and Shorts.

Rules:
- Each clip must be between {MIN_LEN} and {MAX_LEN} seconds long.
- Start at the beginning of a sentence with a strong hook, and end at the end of a complete thought.
- Clips must not overlap.
- Use the timestamps from the transcript (seconds).

Return ONLY a JSON array, no other text, like:
[{{"start": 12.5, "end": 48.0, "title": "Short catchy title"}}]

Transcript:
{transcript}
"""
    api = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
    }
    last = None
    for attempt in range(5):
        r = requests.post(api, headers={"x-goog-api-key": KEY}, json=body, timeout=180)
        if r.status_code in (429, 500, 503):
            last = r.text[:300]
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        return json.loads(text)
    raise RuntimeError(f"Gemini kept failing: {last}")


def validate(picks, duration):
    clips = []
    for p in picks:
        try:
            start = max(0.0, float(p["start"]))
            end = min(float(p["end"]), duration)
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < 8:
            continue
        if end - start > MAX_LEN + 15:
            end = start + MAX_LEN
        clips.append({"start": start, "end": end, "title": str(p.get("title", "clip"))})
    clips.sort(key=lambda c: c["start"])
    result = []
    for c in clips:
        if result and c["start"] < result[-1]["end"]:
            continue
        result.append(c)
    return result[:NUM]


def fmt_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def write_srt(words, start, end, path):
    chunk = [w for w in words if w["start"] >= start - 0.05 and w["end"] <= end + 0.05]
    lines = []
    i = 1
    for k in range(0, len(chunk), 4):
        group = chunk[k:k + 4]
        t0 = group[0]["start"] - start
        t1 = group[-1]["end"] - start
        text = " ".join(w["word"] for w in group).upper()
        lines.append(f"{i}\n{fmt_time(t0)} --> {fmt_time(t1)}\n{text}\n")
        i += 1
    path.write_text("\n".join(lines), encoding="utf-8")
    return bool(lines)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "clip"


def cut(src, clip, words, idx):
    srt = WORK / f"clip{idx:02d}.srt"
    has_subs = write_srt(words, clip["start"], clip["end"], srt)
    out = OUT / f"clip_{idx:02d}_{slug(clip['title'])}.mp4"
    vf = "crop=trunc(ih*9/16/2)*2:ih,scale=1080:1920"
    if has_subs:
        style = "FontSize=11,Bold=1,Outline=2,Shadow=0,Alignment=2,MarginV=30"
        vf += f",subtitles={srt.as_posix()}:force_style='{style}'"
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip['start']:.2f}",
        "-i", str(src),
        "-t", f"{clip['end'] - clip['start']:.2f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def main():
    print("Downloading video...")
    src = download()
    print("Transcribing...")
    segs, words, duration = transcribe(src)
    if not segs:
        raise RuntimeError("No speech found in the video.")
    print(f"Transcribed {len(segs)} segments, {duration:.0f}s total")
    print("Asking Gemini for the best moments...")
    clips = validate(ask_gemini(segs, duration), duration)
    if not clips:
        raise RuntimeError("Gemini returned no usable clips.")
    meta = []
    for i, clip in enumerate(clips, 1):
        print(f"Cutting clip {i}/{len(clips)}: {clip['title']}")
        out = cut(src, clip, words, i)
        meta.append({**clip, "file": out.name})
    (OUT / "clips.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. {len(meta)} clips saved to {OUT}/")


if __name__ == "__main__":
    main()
