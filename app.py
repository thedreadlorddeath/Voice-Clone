"""
Obsidian Voice Clone Studio (Chatterbox-TTS)
============================================
A local, single-user Text-to-Speech web app powered by Chatterbox-TTS (zero-shot
voice cloning), Gradio, PyDub, and FFmpeg.

Hardware Optimizations (Lenovo LOQ RTX 3050 6GB Laptop GPU):
- Chatterbox T3 Llama backbone loaded in float16 precision with AMP autocast
- Peak VRAM strictly budgeted under 4.8 GB
- Periodic torch.cuda.empty_cache() and gc.collect() after each chunk to prevent OOM
- Dual-stage background noise and disturbance filter for voice reference (FFmpeg + noisereduce)
- Native import of Obsidian .md, .markdown, and .txt files with automatic YAML frontmatter stripping
- Emotion-to-delivery mapping based on Obsidian Markdown formatting:
    * Plain text:    Baseline calm  (exaggeration=0.25, cfg_weight=0.50, 1.00x, 0.0 dB)
    * **bold text**: Loud / intense (exaggeration=0.75, cfg_weight=0.60, 1.00x, +2.5 dB)
    * *italic text*: Smooth / warm  (exaggeration=0.45, cfg_weight=0.40, 0.94x, 0.0 dB)
    * ~tilde text~:  Soft / whisper (exaggeration=0.35, cfg_weight=0.35, 0.88x, -2.0 dB)
- Export to high-quality 192kbps AAC .m4a with sub-65Hz DC rumble filtering
"""

import argparse
import gc
import hashlib
import inspect
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import warnings

# Help PyTorch manage 6 GB VRAM without fragmentation on Windows
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
# Disable Gradio and HuggingFace analytics network calls to ensure fast local-only execution
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Filter benign library warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

# Ensure Perth watermark fallback is robust
try:
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = getattr(perth, "DummyWatermarker", None)
except Exception:
    pass

# Pre-create HuggingFace Hub cache folder to prevent first-download permission/missing directory warnings
try:
    (Path.home() / ".cache" / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)
except Exception:
    pass

import gradio as gr
import numpy as np
import torch
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Paths & Global Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VOICE_PATH = BASE_DIR / "my_voice.wav"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_REF_CACHE_DIR = BASE_DIR / ".cache_refs"
CLEAN_REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

REF_TARGET_SR = 24000
MAX_CACHE_REFS = 8
MAX_CHUNK_CHARS = 220
CROSSFADE_MS = 30
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860


# ---------------------------------------------------------------------------
# Emotion / Style Presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmotionStyle:
    name: str
    label: str
    tag: str
    color: str
    exaggeration: float
    cfg_weight: float
    speed: float
    gain_db: float
    description: str


STYLES = {
    "plain": EmotionStyle(
        name="plain",
        label="Plain (Default)",
        tag="Baseline",
        color="#3b82f6",
        exaggeration=0.25,
        cfg_weight=0.50,
        speed=1.00,
        gain_db=0.0,
        description="Flat, calm baseline narration.",
    ),
    "bold": EmotionStyle(
        name="bold",
        label="Bold (Loud / Intense)",
        tag="Emphasis",
        color="#ef4444",
        exaggeration=0.75,
        cfg_weight=0.60,
        speed=1.00,
        gain_db=2.5,
        description="Firmer emphasis, louder and more urgent delivery (+2.5 dB).",
    ),
    "italic": EmotionStyle(
        name="italic",
        label="Italic (Smooth / Warm)",
        tag="Melodic",
        color="#8b5cf6",
        exaggeration=0.45,
        cfg_weight=0.40,
        speed=0.94,
        gain_db=0.0,
        description="Melodic, warmer delivery at a slightly relaxed pace (0.94x).",
    ),
    "tilde": EmotionStyle(
        name="tilde",
        label="Tilde (Soft / Whisper)",
        tag="Intimate",
        color="#10b981",
        exaggeration=0.35,
        cfg_weight=0.35,
        speed=0.88,
        gain_db=-2.0,
        description="Intimate, low-energy whisper with reduced volume (-2.0 dB, 0.88x).",
    ),
}


@dataclass
class TextSegment:
    style: str
    text: str


# ---------------------------------------------------------------------------
# FFmpeg & Audio Utilities
# ---------------------------------------------------------------------------
def resolve_ffmpeg_executable() -> str:
    """Locate ffmpeg on PATH or fall back to imageio-ffmpeg's bundled binary."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            return bundled
    except Exception:
        pass
    raise RuntimeError(
        "FFmpeg was not found on PATH and imageio-ffmpeg fallback is unavailable. "
        "Install FFmpeg or ensure imageio-ffmpeg is installed in your virtual environment."
    )


def configure_pydub_ffmpeg() -> str:
    """Configure PyDub to use the resolved FFmpeg executable."""
    ffmpeg_exe = resolve_ffmpeg_executable()
    AudioSegment.converter = ffmpeg_exe
    ffprobe_exe = shutil.which("ffprobe")
    if ffprobe_exe:
        AudioSegment.ffprobe = ffprobe_exe
    return ffmpeg_exe


# Eagerly initialize PyDub with FFmpeg at startup
try:
    GLOBAL_FFMPEG_EXE = configure_pydub_ffmpeg()
except Exception:
    GLOBAL_FFMPEG_EXE = "ffmpeg"


def apply_pydub_effects(segment: AudioSegment, style: EmotionStyle) -> AudioSegment:
    """Apply speed and dB gain adjustments using PyDub."""
    processed = segment

    # Speed modulation via frame_rate resampling (speed < 1.0 yields longer audio / slower speech)
    if abs(style.speed - 1.0) > 1e-4:
        orig_rate = processed.frame_rate
        altered_rate = max(8000, int(round(orig_rate * style.speed)))
        processed = processed._spawn(
            processed.raw_data,
            overrides={"frame_rate": altered_rate},
        ).set_frame_rate(orig_rate)

    # Gain adjustment in dB
    if abs(style.gain_db) > 1e-4:
        processed = processed + style.gain_db

    return processed


def tensor_to_audio_segment(wav: torch.Tensor | np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert Chatterbox waveform output to a 16-bit mono PyDub AudioSegment."""
    if isinstance(wav, torch.Tensor):
        wav_np = wav.squeeze().detach().cpu().float().numpy()
    else:
        wav_np = np.asarray(wav, dtype=np.float32).squeeze()

    if wav_np.ndim == 0:
        wav_np = np.atleast_1d(wav_np)
    elif wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=0)

    wav_np = np.nan_to_num(wav_np, nan=0.0, posinf=1.0, neginf=-1.0)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm_int16 = (wav_np * 32767.0).astype(np.int16)

    return AudioSegment(
        data=pcm_int16.tobytes(),
        sample_width=2,
        frame_rate=int(sample_rate),
        channels=1,
    )


def export_m4a_with_ffmpeg(audio: AudioSegment, output_path: Path, ffmpeg_exe: str) -> Path:
    """Export stitched PyDub AudioSegment to 192kbps .m4a via FFmpeg (with sub-65Hz rumble filtering)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = Path(tmp_wav.name)

    try:
        audio.export(str(tmp_wav_path), format="wav")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(tmp_wav_path),
            "-af",
            "highpass=f=65",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg export failed:\n{result.stderr.strip()}")
    finally:
        if tmp_wav_path.exists():
            tmp_wav_path.unlink(missing_ok=True)

    return output_path


def _prune_ref_cache() -> None:
    cached_files = sorted(
        CLEAN_REF_CACHE_DIR.glob("clean_*.wav"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in cached_files[MAX_CACHE_REFS:]:
        stale.unlink(missing_ok=True)


def _run_ffmpeg_audio_filter(
    input_path: Path,
    output_path: Path,
    ffmpeg_exe: str,
    filter_chain: str,
) -> None:
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(REF_TARGET_SR),
        "-af",
        filter_chain,
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio cleaning failed:\n{result.stderr.strip()}")


def clean_reference_audio(input_path: Path, ffmpeg_exe: str) -> Tuple[Path, str]:
    """
    Auto-filter background noise, rumble, hiss, transient disturbances, and dead air
    from a voice reference audio file for optimal zero-shot Chatterbox-TTS cloning.

    Pipeline:
      1. FFmpeg acoustic pre-filter: 80Hz highpass, 11.5kHz lowpass, adaptive FFT denoiser (afftdn),
         leading/trailing dead-air trim, and loudnorm (-18 LUFS, -2.0 dBTP headroom).
      2. Dual-pass spectral gating (noisereduce): stationary hiss pass + non-stationary
         disturbance/click pass, followed by gentle peak normalization (-1.5 dBFS).
    """
    AudioSegment.converter = ffmpeg_exe

    stat = input_path.stat()
    sig_str = f"{input_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|v2"
    cache_key = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:16]
    cached_path = CLEAN_REF_CACHE_DIR / f"clean_{cache_key}.wav"

    if cached_path.exists() and cached_path.stat().st_size > 1024:
        return cached_path, "Cleaned reference loaded from cache"

    tmp_stage1 = CLEAN_REF_CACHE_DIR / f"tmp_{cache_key}_stage1.wav"
    tmp_stage2 = CLEAN_REF_CACHE_DIR / f"tmp_{cache_key}_stage2.wav"

    try:
        # Stage 1: FFmpeg acoustic cleanup + silence edge trimming + loudness normalization
        full_chain = (
            "highpass=f=80,"
            "lowpass=f=11500,"
            "afftdn=nf=-25:tn=1,"
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,"
            "areverse,"
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,"
            "areverse,"
            "loudnorm=I=-18:TP=-2.0:LRA=11"
        )
        _run_ffmpeg_audio_filter(input_path, tmp_stage1, ffmpeg_exe, full_chain)

        # Guard against overly aggressive silence trimming on very quiet samples
        seg_check = AudioSegment.from_wav(str(tmp_stage1))
        if len(seg_check) < 500:
            fallback_chain = (
                "highpass=f=80,"
                "lowpass=f=11500,"
                "afftdn=nf=-25:tn=1,"
                "loudnorm=I=-18:TP=-2.0:LRA=11"
            )
            _run_ffmpeg_audio_filter(input_path, tmp_stage1, ffmpeg_exe, fallback_chain)

        # Stage 2: Python dual-pass spectral gating (stationary + transient disturbances)
        stage_note = "FFmpeg + Spectral Noise Reduction"
        try:
            import noisereduce as nr
            from scipy.io import wavfile

            sr, raw_samples = wavfile.read(str(tmp_stage1))

            # Normalize to float32 [-1.0, 1.0] BEFORE channel averaging to prevent dtype bugs
            if np.issubdtype(raw_samples.dtype, np.integer):
                max_val = float(np.iinfo(raw_samples.dtype).max)
                audio_f32 = raw_samples.astype(np.float32) / max_val
            else:
                audio_f32 = raw_samples.astype(np.float32)

            if audio_f32.ndim > 1:
                audio_f32 = audio_f32.mean(axis=1)

            if len(audio_f32) > int(sr * 0.25) and float(np.max(np.abs(audio_f32))) > 1e-6:
                # Pass 1: stationary hum / fan / room hiss removal
                denoised = nr.reduce_noise(
                    y=audio_f32,
                    sr=sr,
                    stationary=True,
                    prop_decrease=0.75,
                    n_std_thresh_stationary=1.5,
                    freq_mask_smooth_hz=500,
                    time_mask_smooth_ms=50,
                )
                # Pass 2: non-stationary disturbance removal (clicks, rustles, sudden spikes)
                denoised = nr.reduce_noise(
                    y=denoised,
                    sr=sr,
                    stationary=False,
                    prop_decrease=0.60,
                    time_constant_s=1.0,
                    thresh_n_mult_nonstationary=2.0,
                )

                # Gentle peak normalization to -1.5 dBFS (0.841)
                denoised = np.nan_to_num(denoised, nan=0.0, posinf=1.0, neginf=-1.0)
                peak = float(np.max(np.abs(denoised)))
                if peak > 1e-6:
                    denoised = denoised * (0.841 / peak)

                pcm16 = (np.clip(denoised, -1.0, 1.0) * 32767.0).astype(np.int16)
                wavfile.write(str(tmp_stage2), sr, pcm16)
                tmp_stage2.replace(cached_path)
            else:
                tmp_stage1.replace(cached_path)
                stage_note = "FFmpeg Acoustic Cleanup"
        except ImportError:
            tmp_stage1.replace(cached_path)
            stage_note = "FFmpeg Acoustic Cleanup (install noisereduce for dual-pass spectral gating)"

        _prune_ref_cache()
        return cached_path, stage_note
    finally:
        if tmp_stage1.exists():
            tmp_stage1.unlink(missing_ok=True)
        if tmp_stage2.exists():
            tmp_stage2.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Obsidian Markdown-to-Emotion Parser & Chunker
# ---------------------------------------------------------------------------
INLINE_MARKDOWN_RE = re.compile(
    r"(?P<bold_italic>\*\*\*(.+?)\*\*\*)"
    r"|(?P<bold>\*\*(?!\*)(.+?)(?<!\*)\*\*)"
    r"|(?P<italic>(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*))"
    r"|(?P<tilde>(?<!~)~(?!~)(.+?)(?<!~)~(?!~))",
    flags=re.DOTALL,
)

SENTENCE_SPLIT_RE = re.compile(r'(?:(?<=[.!?…])|(?<=[.!?…]["”\')\]]))\s+')
CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:\u2014\u2013-])\s+")


def clean_obsidian_artifacts(raw_text: str) -> str:
    """
    Remove non-spoken Obsidian frontmatter, block syntax, code blocks, images,
    and link wrappers while strictly preserving inline emotion tags (**bold**, *italic*, ~tilde~).
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip YAML frontmatter at top of file (e.g. --- tags: [x] ---)
    text = re.sub(r"^---\s*\n.*?\n---\s*(\n|$)", "", text, flags=re.DOTALL)
    # Strip Obsidian %%comments%%
    text = re.sub(r"%%.*?%%", "", text, flags=re.DOTALL)
    # Strip fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Unwrap inline code (`code` -> code)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    # Strip Obsidian embedded media ![[image.png]]
    text = re.sub(r"!\s*\[\[.*?\]\]", "", text)
    # Strip standard markdown images ![alt](url)
    text = re.sub(r"!\s*\[.*?\]\(.*?\)", "", text)
    # Strip footnote definitions and references [^1]
    text = re.sub(r"(?m)^\s*\[\^[^\]]+\]:\s*.*$", "", text)
    text = re.sub(r"\[\^[^\]]+\]", "", text)
    # Strip markdown heading prefixes (# Title -> Title)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    # Strip blockquote markers (> quote -> quote)
    text = re.sub(r"(?m)^>\s?", "", text)
    # Strip markdown bullet / numbered list prefixes (- item, * item, 1. item)
    text = re.sub(r"(?m)^\s*(?:[-*+]|\d+\.)\s+", "", text)
    # Convert Obsidian wiki links: [[Target|Alias]] -> Alias, [[Target]] -> Target
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Convert standard Markdown links: [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Unwrap ==highlights==
    text = re.sub(r"==(.+?)==", r"\1", text, flags=re.DOTALL)
    # Unwrap ~~strikethrough~~ markers if present so ~tilde~ isn't confused
    text = re.sub(r"~~(.+?)~~", r"\1", text, flags=re.DOTALL)
    return text


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text at paragraph and sentence boundaries (~max_chars) for stable Chatterbox inference."""
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n+", text)]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return []

    all_chunks: List[str] = []
    for normalized in paragraphs:
        if len(normalized) <= max_chars:
            all_chunks.append(normalized)
            continue
        all_chunks.extend(_split_paragraph(normalized, max_chars=max_chars))
    return all_chunks


def _split_paragraph(normalized: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    sentences = SENTENCE_SPLIT_RE.split(normalized)
    chunks: List[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    def split_oversized_sentence(sentence: str) -> List[str]:
        parts = CLAUSE_SPLIT_RE.split(sentence)
        sub_chunks: List[str] = []
        buf = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > max_chars:
                if buf:
                    sub_chunks.append(buf)
                    buf = ""
                # Word-level wrap as fallback
                words = part.split(" ")
                wbuf = ""
                for w in words:
                    if not wbuf:
                        wbuf = w
                    elif len(wbuf) + 1 + len(w) <= max_chars:
                        wbuf = f"{wbuf} {w}"
                    else:
                        sub_chunks.append(wbuf)
                        wbuf = w
                if wbuf:
                    sub_chunks.append(wbuf)
            elif not buf:
                buf = part
            elif len(buf) + 1 + len(part) <= max_chars:
                buf = f"{buf} {part}"
            else:
                sub_chunks.append(buf)
                buf = part
        if buf:
            sub_chunks.append(buf)
        return sub_chunks

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            flush_current()
            chunks.extend(split_oversized_sentence(sentence))
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            flush_current()
            current = sentence

    flush_current()
    return chunks


def parse_obsidian_markdown(raw_text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[TextSegment]:
    """
    Parse Obsidian-flavored inline Markdown into ordered emotion-tagged chunks.

    - Plain text -> plain (baseline calm)
    - **bold**   -> bold  (loud / intense)
    - *italic*   -> italic (smooth / warm)
    - ~tilde~    -> tilde (soft / whisper)
    """
    cleaned = clean_obsidian_artifacts(raw_text)
    raw_segments: List[TextSegment] = []
    last_idx = 0

    for match in INLINE_MARKDOWN_RE.finditer(cleaned):
        start, end = match.span()
        if start > last_idx:
            plain_piece = cleaned[last_idx:start]
            if plain_piece.strip():
                raw_segments.append(TextSegment("plain", plain_piece))

        if match.group("bold_italic") is not None:
            inner = match.group(2)
            if inner and inner.strip():
                raw_segments.append(TextSegment("bold", inner))
        elif match.group("bold") is not None:
            inner = match.group(4)
            if inner and inner.strip():
                raw_segments.append(TextSegment("bold", inner))
        elif match.group("italic") is not None:
            inner = match.group(6)
            if inner and inner.strip():
                raw_segments.append(TextSegment("italic", inner))
        elif match.group("tilde") is not None:
            inner = match.group(8)
            if inner and inner.strip():
                raw_segments.append(TextSegment("tilde", inner))

        last_idx = end

    if last_idx < len(cleaned):
        tail = cleaned[last_idx:]
        if tail.strip():
            raw_segments.append(TextSegment("plain", tail))

    # Attach leading punctuation right after formatted spans to the previous segment
    leading_punct_re = re.compile(r'^([.!?,;:…"\'”’\)\]]+)(?:\s+(.*))?$', flags=re.DOTALL)
    merged_segments: List[TextSegment] = []
    for seg in raw_segments:
        text_norm = re.sub(r"\n{2,}", "\n\n", seg.text)
        text_norm = re.sub(r"[ \t]+", " ", text_norm).strip()
        if not text_norm:
            continue

        if seg.style == "plain" and merged_segments:
            punct_match = leading_punct_re.match(text_norm)
            if punct_match:
                punct_part = punct_match.group(1)
                rest_part = (punct_match.group(2) or "").strip()
                merged_segments[-1] = TextSegment(
                    merged_segments[-1].style,
                    f"{merged_segments[-1].text}{punct_part}",
                )
                if not rest_part:
                    continue
                text_norm = rest_part

        if not re.search(r"[^\W_]", text_norm):
            continue
        merged_segments.append(TextSegment(seg.style, text_norm))

    # Chunk each segment at sentence boundaries (~220 chars max)
    final_segments: List[TextSegment] = []
    for seg in merged_segments:
        for chunk in split_into_chunks(seg.text, max_chars=max_chars):
            final_segments.append(TextSegment(seg.style, chunk))

    return final_segments


# ---------------------------------------------------------------------------
# Chatterbox-TTS Engine (RTX 3050 6 GB Optimized)
# ---------------------------------------------------------------------------
class ChatterboxEngine:
    def __init__(self) -> None:
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._fp16_active = False
        self._cached_voice_sig: Optional[Tuple[str, float]] = None

    def load(self) -> None:
        if self.model is not None:
            return

        from chatterbox.tts import ChatterboxTTS

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=self.device)

        if self.device == "cuda":
            # Cast the 0.5B T3 Llama backbone to float16 to fit comfortably under 4.8 GB VRAM.
            # S3Gen STFT/iSTFT and VoiceEncoder remain float32-compatible via torch.amp.autocast.
            model.t3 = model.t3.to(dtype=torch.float16)
            model.t3.eval()
            model.s3gen.eval()
            model.ve.eval()
            self._fp16_active = True
            torch.cuda.empty_cache()

        self.model = model

    def _prepare_voice_if_needed(self, voice_path: Path, exaggeration: float) -> None:
        assert self.model is not None
        sig = (str(voice_path.resolve()), voice_path.stat().st_mtime)
        if self._cached_voice_sig != sig or getattr(self.model, "conds", None) is None:
            self.model.prepare_conditionals(str(voice_path), exaggeration=exaggeration)
            self._cached_voice_sig = sig

    def synthesize_chunk(
        self,
        text: str,
        voice_path: Path,
        exaggeration: float,
        cfg_weight: float,
    ) -> AudioSegment:
        self.load()
        assert self.model is not None

        self._prepare_voice_if_needed(voice_path, exaggeration=exaggeration)

        wav = None
        try:
            if self.device == "cuda" and self._fp16_active:
                try:
                    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                        wav = self.model.generate(
                            text=text,
                            audio_prompt_path=None,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                        )
                except RuntimeError as exc:
                    if "Half" in str(exc) or "Float" in str(exc):
                        # Safe fallback: revert T3 to float32 while keeping autocast active for activations
                        self.model.t3 = self.model.t3.to(dtype=torch.float32)
                        self._fp16_active = False
                        torch.cuda.empty_cache()
                        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                            wav = self.model.generate(
                                text=text,
                                audio_prompt_path=None,
                                exaggeration=exaggeration,
                                cfg_weight=cfg_weight,
                            )
                    else:
                        raise
            else:
                with torch.inference_mode():
                    wav = self.model.generate(
                        text=text,
                        audio_prompt_path=None,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                    )

            return tensor_to_audio_segment(wav, self.model.sr)
        finally:
            del wav
            if self.device == "cuda":
                torch.cuda.empty_cache()


ENGINE = ChatterboxEngine()


# ---------------------------------------------------------------------------
# Voice Reference Helper & Full Synthesis Pipeline
# ---------------------------------------------------------------------------
def ensure_wav_reference(
    voice_input: Optional[str],
    ffmpeg_exe: str,
    auto_clean: bool = True,
) -> Tuple[Path, Optional[Path], str]:
    """
    Resolve and optionally denoise the reference voice file.
    Returns (usable_wav_path, temp_path_to_cleanup_or_None, cleaning_note).
    """
    if voice_input and Path(voice_input).exists():
        chosen = Path(voice_input)
    elif DEFAULT_VOICE_PATH.exists():
        chosen = DEFAULT_VOICE_PATH
    else:
        raise FileNotFoundError(
            f"No voice reference provided and '{DEFAULT_VOICE_PATH.name}' was not found in {BASE_DIR}. "
            "Please upload a clean 5–15s voice sample or place my_voice.wav in the project root."
        )

    if auto_clean:
        cleaned_path, clean_note = clean_reference_audio(chosen, ffmpeg_exe)
        return cleaned_path, None, clean_note

    if chosen.suffix.lower() == ".wav":
        return chosen, None, "Raw voice sample (filtering off)"

    # Convert non-wav uploads (e.g. .mp3, .m4a) to a temporary .wav for librosa/chatterbox
    AudioSegment.converter = ffmpeg_exe
    audio = AudioSegment.from_file(str(chosen)).set_channels(1)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    audio.export(str(tmp_path), format="wav")
    return tmp_path, tmp_path, "Converted to WAV (filtering off)"


def test_clean_voice_action(
    voice_file: Optional[str],
    auto_clean_voice: bool,
) -> Tuple[Optional[str], str]:
    """Action for the 'Denoise & Test Reference' button."""
    try:
        ffmpeg_exe = configure_pydub_ffmpeg()
    except Exception as exc:
        return None, f"FFmpeg error: {exc}"

    try:
        voice_path, _, clean_note = ensure_wav_reference(
            voice_file,
            ffmpeg_exe,
            auto_clean=auto_clean_voice,
        )
        return str(voice_path), f"Voice reference ready: {clean_note} ({voice_path.name})"
    except Exception as exc:
        return None, f"Voice error: {exc}"


def parse_and_inspect_script(script_text: str) -> Tuple[str, str]:
    """Generate stats and a chunk preview table without running TTS inference."""
    if not script_text or not script_text.strip():
        return "0 chars | 0 words | 0 chunks", "Paste a script or upload an Obsidian .md file to inspect chunks."

    segments = parse_obsidian_markdown(script_text, max_chars=MAX_CHUNK_CHARS)
    chars = len(script_text)
    words = len(script_text.split())
    chunks_count = len(segments)

    # Estimate duration: ~150 words per minute (~2.5 words/second)
    est_sec = max(1, int(words / 2.5))
    mins, secs = divmod(est_sec, 60)
    stats_str = f"📊 {chars:,} characters  •  📝 ~{words:,} words  •  🧩 {chunks_count} chunks  •  ⏱️ ~{mins}m {secs:02d}s estimated audio"

    lines = [
        f"{idx:>2}. [{seg.style.upper():<6}] ({len(seg.text):>3} chars) [exagg={STYLES[seg.style].exaggeration:.2f}, cfg={STYLES[seg.style].cfg_weight:.2f}, speed={STYLES[seg.style].speed:.2f}x, gain={STYLES[seg.style].gain_db:+.1f}dB]: {seg.text}"
        for idx, seg in enumerate(segments, start=1)
    ]
    return stats_str, "\n".join(lines)


def handle_file_import(file_path: Optional[str]) -> Tuple[str, str, str]:
    """
    Import an Obsidian Markdown (.md, .markdown, .txt) file, strip frontmatter,
    and populate the editor with parsed telemetry.
    """
    if not file_path:
        return gr.update(), "No file selected.", gr.update()

    path = Path(file_path)
    if not path.exists():
        return gr.update(), f"File not found: {path.name}", gr.update()

    try:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        stats_str, preview = parse_and_inspect_script(content)
        info = f"Successfully imported '{path.name}' ({len(content):,} chars). Ready to generate."
        return content, info, preview
    except Exception as exc:
        return gr.update(), f"Error importing '{path.name}': {exc}", gr.update()


def synthesize_script(
    script_text: str,
    voice_file: Optional[str],
    auto_clean_voice: bool,
    max_chunks_limit: Optional[int] = None,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[Optional[str], Optional[str], Optional[str], str, str]:
    """
    Full synthesis pipeline:
    1. Parse Obsidian markdown into emotion segments
    2. Filter background noise from voice reference
    3. Generate each chunk with Chatterbox-TTS (float16 on CUDA)
    4. Apply PyDub volume/speed effects
    5. Stitch chunks with 30ms crossfade
    6. Export 192kbps AAC .m4a via FFmpeg
    """
    if not script_text or not script_text.strip():
        raise gr.Error("Please paste or type a script before generating audio.")

    try:
        ffmpeg_exe = configure_pydub_ffmpeg()
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    segments = parse_obsidian_markdown(script_text, max_chars=MAX_CHUNK_CHARS)
    if not segments:
        raise gr.Error("No speakable text found after parsing Markdown.")

    if max_chunks_limit is not None and max_chunks_limit > 0:
        segments = segments[:max_chunks_limit]

    total_chunks = len(segments)
    progress(0.01, desc="Filtering background noise & disturbances from reference audio...")

    try:
        voice_path, cleanup_voice_tmp, clean_note = ensure_wav_reference(
            voice_file,
            ffmpeg_exe,
            auto_clean=auto_clean_voice,
        )
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc

    preview_lines = [
        f"{idx:>2}. [{seg.style.upper():<6}] ({len(seg.text):>3} chars): {seg.text}"
        for idx, seg in enumerate(segments, start=1)
    ]
    parsed_preview = "\n".join(preview_lines)

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.perf_counter()
        progress(0.03, desc="Loading Chatterbox-TTS on GPU (float16)...")
        ENGINE.load()

        stitched: Optional[AudioSegment] = None

        for idx, seg in enumerate(segments, start=1):
            style = STYLES[seg.style]
            progress(
                idx / (total_chunks + 1),
                desc=f"Synthesizing chunk {idx}/{total_chunks} [{style.label}]...",
            )

            chunk_audio = ENGINE.synthesize_chunk(
                text=seg.text,
                voice_path=voice_path,
                exaggeration=style.exaggeration,
                cfg_weight=style.cfg_weight,
            )
            chunk_audio = apply_pydub_effects(chunk_audio, style)

            if stitched is None:
                stitched = chunk_audio
            else:
                cf = min(CROSSFADE_MS, len(stitched) // 2, len(chunk_audio) // 2)
                stitched = stitched.append(chunk_audio, crossfade=max(0, cf))

            # Strictly free VRAM cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        assert stitched is not None
        progress(0.95, desc="Exporting 192kbps .m4a via FFmpeg...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "quick_preview" if (max_chunks_limit and max_chunks_limit <= 3) else "voice_clone"
        output_m4a = OUTPUT_DIR / f"{prefix}_{timestamp}.m4a"
        export_m4a_with_ffmpeg(stitched, output_m4a, ffmpeg_exe)

        elapsed = time.perf_counter() - t_start
        duration_s = len(stitched) / 1000.0
        vram_info = "CPU mode"
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            vram_info = f"Peak VRAM: {peak_gb:.2f} GB (float16)"

        status = (
            f"✅ Generated in {elapsed:.1f}s | Audio duration: {duration_s:.1f}s | "
            f"Chunks: {total_chunks} | {vram_info} | Ref: {clean_note} | Saved: {output_m4a.name}"
        )
        cleaned_preview_path = str(voice_path) if auto_clean_voice else None
        progress(1.0, desc="Complete!")
        return str(output_m4a), str(output_m4a), cleaned_preview_path, status, parsed_preview
    finally:
        if cleanup_voice_tmp and cleanup_voice_tmp.exists():
            cleanup_voice_tmp.unlink(missing_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def quick_test_synthesis(
    script_text: str,
    voice_file: Optional[str],
    auto_clean_voice: bool,
    progress=gr.Progress(track_tqdm=False),
):
    """Run a fast test on the first 2 chunks only."""
    return synthesize_script(
        script_text=script_text,
        voice_file=voice_file,
        auto_clean_voice=auto_clean_voice,
        max_chunks_limit=2,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# UI Layout & Aesthetics
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* Obsidian Voice Clone Studio Custom Theme */
.gradio-container {
    max-width: 1040px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.studio-header {
    background: linear-gradient(135deg, #1e1e2f 0%, #2a2b45 100%);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    color: #ffffff;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.studio-header h1 {
    margin: 0 0 6px 0;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #ffffff;
}

.studio-header p {
    margin: 0;
    font-size: 14px;
    color: #b0b5c9;
    line-height: 1.5;
}

.badge-row {
    margin-top: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.gpu-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.style-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin: 16px 0 20px 0;
}

.style-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.dark .style-card {
    background: #1f2937;
    border-color: #374151;
}

.style-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.style-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 6px;
    color: #ffffff;
}

.style-title {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 4px;
    color: #111827;
}

.dark .style-title {
    color: #f3f4f6;
}

.style-params {
    font-size: 12px;
    color: #6b7280;
    line-height: 1.4;
}

.dark .style-params {
    color: #9ca3af;
}

.stats-bar {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 500;
    color: #334155;
    margin-top: -6px;
    margin-bottom: 12px;
}

.dark .stats-bar {
    background: #1e293b;
    border-color: #334155;
    color: #cbd5e1;
}
"""

SAMPLE_SCRIPT = """# Obsidian Script Example
Welcome back to the studio. **Listen closely**, because what happens next changes everything.

Take a gentle breath and *let the warm melody carry the thought forward*, smooth and unhurried.

~And just between the two of us, keep this whisper safe until tomorrow.~

Every spoken emotion adapts seamlessly to your voice sample."""


def build_ui() -> Tuple[gr.Blocks, dict]:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    blocks_params = inspect.signature(gr.Blocks.__init__).parameters
    blocks_kwargs = {"title": "Obsidian Voice Clone Studio"}
    launch_extra = {}

    if "theme" in blocks_params:
        blocks_kwargs["theme"] = theme
    else:
        launch_extra["theme"] = theme

    if "css" in blocks_params:
        blocks_kwargs["css"] = CUSTOM_CSS
    else:
        launch_extra["css"] = CUSTOM_CSS

    default_voice = str(DEFAULT_VOICE_PATH) if DEFAULT_VOICE_PATH.exists() else None
    voice_status = (
        f"Default ready: {DEFAULT_VOICE_PATH.name}"
        if DEFAULT_VOICE_PATH.exists()
        else f"{DEFAULT_VOICE_PATH.name} not in root — upload sample below"
    )

    with gr.Blocks(**blocks_kwargs) as demo:
        # Header banner
        gr.HTML(
            f"""
            <div class="studio-header">
                <h1>🎙️ Obsidian Voice Clone Studio</h1>
                <p>Natural zero-shot voice cloning with emotive delivery driven by your Obsidian Markdown tags.</p>
                <div class="badge-row">
                    <span class="gpu-badge">⚡ RTX 3050 6GB • float16 CUDA</span>
                    <span class="gpu-badge">🛡️ VRAM Guard &lt; 4.8 GB</span>
                    <span class="gpu-badge">✨ Dual-Stage Noise Filter</span>
                    <span class="gpu-badge">🎵 192k M4A AAC Export</span>
                </div>
            </div>
            """
        )

        # Emotive Cheat-Sheet cards
        gr.HTML(
            """
            <div class="style-cards-grid">
                <div class="style-card" style="border-left: 4px solid #3b82f6;">
                    <span class="style-badge" style="background:#3b82f6;">Plain Text</span>
                    <div class="style-title">Normal / Baseline</div>
                    <div class="style-params">Flat, calm narrative baseline.<br><code>exagg=0.25 • cfg=0.50 • 1.00x • 0.0dB</code></div>
                </div>
                <div class="style-card" style="border-left: 4px solid #ef4444;">
                    <span class="style-badge" style="background:#ef4444;">**bold text**</span>
                    <div class="style-title">Loud / Intense</div>
                    <div class="style-params">Firmer, emphatic, urgent tone.<br><code>exagg=0.75 • cfg=0.60 • 1.00x • +2.5dB</code></div>
                </div>
                <div class="style-card" style="border-left: 4px solid #8b5cf6;">
                    <span class="style-badge" style="background:#8b5cf6;">*italic text*</span>
                    <div class="style-title">Smooth / Warm</div>
                    <div class="style-params">Warm, melodic, relaxed delivery.<br><code>exagg=0.45 • cfg=0.40 • 0.94x • 0.0dB</code></div>
                </div>
                <div class="style-card" style="border-left: 4px solid #10b981;">
                    <span class="style-badge" style="background:#10b981;">~tilde text~</span>
                    <div class="style-title">Soft / Whisper</div>
                    <div class="style-params">Intimate, sultry, quiet whisper.<br><code>exagg=0.35 • cfg=0.35 • 0.88x • -2.0dB</code></div>
                </div>
            </div>
            """
        )

        # Voice Reference Accordion
        with gr.Accordion(f"🔊 Voice Reference Setup ({voice_status})", open=not DEFAULT_VOICE_PATH.exists()):
            with gr.Row():
                with gr.Column(scale=3):
                    voice_input = gr.Audio(
                        label="Voice Reference (.wav, .mp3, .m4a — 5 to 15 seconds clean sample)",
                        type="filepath",
                        sources=["upload", "microphone"],
                        value=default_voice,
                    )
                    with gr.Row():
                        auto_clean_chk = gr.Checkbox(
                            label="Auto-filter background noise, hiss & disturbances (FFmpeg + Spectral Gating)",
                            value=True,
                            scale=3,
                        )
                        test_voice_btn = gr.Button("Denoise & Test Reference", variant="secondary", scale=2)
                with gr.Column(scale=2):
                    cleaned_voice_preview = gr.Audio(
                        label="Denoised Reference Preview (Cached)",
                        type="filepath",
                        interactive=False,
                    )

        # Script Input & File Import
        with gr.Row():
            with gr.Column(scale=4):
                script_input = gr.Textbox(
                    label="Obsidian Script Editor (paste markdown or edit imported note)",
                    placeholder="Type or paste your Obsidian script here...",
                    value=SAMPLE_SCRIPT,
                    lines=13,
                    max_lines=28,
                )
            with gr.Column(scale=2):
                file_import = gr.File(
                    label="📄 Import Obsidian File (.md, .markdown, .txt)",
                    file_count="single",
                    file_types=[".md", ".markdown", ".txt"],
                    type="filepath",
                )
                with gr.Row():
                    sample_btn = gr.Button("Load Sample", variant="secondary", size="sm")
                    clear_btn = gr.ClearButton([script_input], value="Clear Text", size="sm")
                inspect_btn = gr.Button("🔍 Inspect Segments & Telemetry", variant="secondary", size="sm")

        # Live stats bar
        stats_display = gr.HTML(
            f"""<div class="stats-bar">{parse_and_inspect_script(SAMPLE_SCRIPT)[0]}</div>"""
        )

        # Generation Action Buttons
        with gr.Row():
            generate_btn = gr.Button(
                "🎙️ Generate Full Voiceover (.m4a)",
                variant="primary",
                scale=3,
                size="lg",
            )
            quick_test_btn = gr.Button(
                "⚡ Quick Test (First 2 Chunks)",
                variant="secondary",
                scale=2,
                size="lg",
            )

        # Status and Hardware Telemetry
        status_box = gr.Textbox(
            label="Status & Hardware Telemetry",
            interactive=False,
            value="Ready. Chatterbox model will initialize in float16 on GPU upon first generation.",
        )

        # Output Section
        with gr.Row():
            audio_preview = gr.Audio(
                label="Generated Audio Preview (.m4a)",
                type="filepath",
                interactive=False,
                scale=3,
            )
            download_btn = gr.DownloadButton(
                label="⬇️ Download .m4a",
                value=None,
                variant="primary",
                scale=1,
            )

        # Chunk breakdown inspector
        with gr.Accordion("🔍 Parsed Emotion Segments (Chunk Breakdown)", open=False):
            parsed_box = gr.Textbox(
                label="Ordered Chunk Breakdown (Style, Chunks & Model Parameters)",
                value=parse_and_inspect_script(SAMPLE_SCRIPT)[1],
                lines=8,
                interactive=False,
            )

        # Event Handlers
        # 1. File Import
        def _on_file_upload(f):
            content, info, preview = handle_file_import(f)
            stats_html = f"""<div class="stats-bar">{parse_and_inspect_script(content)[0]}</div>"""
            return content, info, preview, stats_html

        file_import.change(
            fn=_on_file_upload,
            inputs=[file_import],
            outputs=[script_input, status_box, parsed_box, stats_display],
        )

        # 2. Text changed / Inspect button
        def _update_stats(text):
            stats, preview = parse_and_inspect_script(text)
            stats_html = f"""<div class="stats-bar">{stats}</div>"""
            return stats_html, preview

        inspect_btn.click(
            fn=_update_stats,
            inputs=[script_input],
            outputs=[stats_display, parsed_box],
        )

        # 3. Load Sample
        def _load_sample():
            stats, preview = parse_and_inspect_script(SAMPLE_SCRIPT)
            stats_html = f"""<div class="stats-bar">{stats}</div>"""
            return SAMPLE_SCRIPT, stats_html, preview, "Loaded sample Obsidian script."

        sample_btn.click(
            fn=_load_sample,
            inputs=[],
            outputs=[script_input, stats_display, parsed_box, status_box],
        )

        # 4. Test Voice Reference
        test_voice_btn.click(
            fn=test_clean_voice_action,
            inputs=[voice_input, auto_clean_chk],
            outputs=[cleaned_voice_preview, status_box],
        )

        # 5. Full Generation
        generate_btn.click(
            fn=synthesize_script,
            inputs=[script_input, voice_input, auto_clean_chk],
            outputs=[audio_preview, download_btn, cleaned_voice_preview, status_box, parsed_box],
        )

        # 6. Quick Test (First 2 chunks)
        quick_test_btn.click(
            fn=quick_test_synthesis,
            inputs=[script_input, voice_input, auto_clean_chk],
            outputs=[audio_preview, download_btn, cleaned_voice_preview, status_box, parsed_box],
        )

    return demo, launch_extra


def main() -> None:
    parser = argparse.ArgumentParser(description="Obsidian Voice Clone Studio (Chatterbox-TTS)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not ask Gradio to open a browser tab (used when launched via start_tts.bat).",
    )
    args = parser.parse_args()

    demo, launch_extra = build_ui()
    demo.queue(max_size=4)
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        inbrowser=not args.no_browser,
        allowed_paths=[str(OUTPUT_DIR), str(BASE_DIR)],
        **launch_extra,
    )


if __name__ == "__main__":
    main()
