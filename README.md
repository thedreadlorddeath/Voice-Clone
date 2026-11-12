# 🎙️ Obsidian Voice Clone Studio (Chatterbox-TTS)

A local, single-user Text-to-Speech web application powered by **Chatterbox-TTS** (zero-shot voice cloning), **Gradio**, **PyDub**, and **FFmpeg**.

Designed specifically for Obsidian writers and storytellers: paste or import your Obsidian notes directly, and inline formatting controls the emotion, pacing, and loudness of your cloned voice in real time.

---

## ✨ Features

- **Zero-Shot Voice Cloning**: Clone any voice using just a 5–15 second clean `.wav`, `.mp3`, or `.m4a` sample.
- **Obsidian Markdown-to-Emotion Parser**: Translates Obsidian inline formatting into distinct emotive delivery styles:
  - `Plain text` (Baseline): Flat, calm narrative baseline.
  - `**bold text**` (Loud / Intense): Firmer emphasis, urgent tone (`+2.5 dB` boost).
  - `*italic text*` (Smooth / Warm): Melodic, relaxed delivery (`0.94x` speed).
  - `~tilde text~` (Soft / Whisper): Intimate, quiet whisper (`0.88x` speed, `-2.0 dB` attenuation).
- **Native Obsidian File Import**: Drag-and-drop or upload `.md`, `.markdown`, or `.txt` files directly. Automatically strips YAML frontmatter (`--- tags: ... ---`), Obsidian comments (`%%...%%`), code blocks, footnotes, and embedded media (`![[...] ]`) so only spoken content is read.
- **Dual-Stage Noise Reduction**: Auto-filters background room hiss, fan hum, transient clicks, and dead air from the reference sample using FFmpeg acoustic pre-filtering and `noisereduce` dual-pass spectral gating with SHA-256 caching.
- **Hardware Optimized for 6 GB VRAM**:
  - T3 Llama backbone loaded in `float16` precision with AMP autocast.
  - Strictly budgets peak VRAM under 4.8 GB on GPUs like the NVIDIA GeForce RTX 3050 6GB Laptop GPU.
  - Automatic `torch.cuda.empty_cache()` and garbage collection after every text chunk to prevent out-of-memory errors on long scripts.
- **Seamless Stitching & Export**: Synthesizes chunks individually, applies PyDub volume/speed adjustments, stitches with 30ms crossfading, and exports to 192kbps AAC `.m4a` via FFmpeg with sub-65Hz rumble filtering.
- **Quick Test Workflow**: Preview the first 2 chunks in seconds before synthesizing large multi-page scripts.

---

## 🎭 Emotion & Delivery Mapping

| Syntax | Delivery Style | Chatterbox Parameters | PyDub Post-Processing |
| :--- | :--- | :--- | :--- |
| `Plain text` | **Calm / Baseline** | `exaggeration=0.25`, `cfg_weight=0.50` | `1.00x` speed, `0.0 dB` gain |
| `**bold text**` | **Loud / Intense** | `exaggeration=0.75`, `cfg_weight=0.60` | `1.00x` speed, `+2.5 dB` boost |
| `*italic text*` | **Smooth / Warm** | `exaggeration=0.45`, `cfg_weight=0.40` | `0.94x` speed, `0.0 dB` gain |
| `~tilde text~` | **Soft / Whisper** | `exaggeration=0.35`, `cfg_weight=0.35` | `0.88x` speed, `-2.0 dB` gain |

---

## 🛠️ Requirements & System Setup

### Prerequisites
- **OS**: Windows 10/11 (or Linux)
- **GPU**: NVIDIA GPU with 6 GB+ VRAM (e.g. RTX 3050 Laptop GPU or higher)
- **CUDA**: 12.4+ supported
- **Python**: Python 3.11 (Official signed Python recommended on Windows)
- **FFmpeg**: Handled automatically via `imageio-ffmpeg` or system PATH

---

## 🚀 Quick Start (Windows)

### 1. Automated Setup
Double-click `setup_env.bat` or run:
```bat
setup_env.bat
```
This script:
1. Installs official signed Python 3.11 (via `winget` if not present).
2. Creates a virtual environment in `.venv`.
3. Installs CUDA-enabled PyTorch (`torch==2.6.0+cu124`).
4. Installs pinned dependencies compatible with Windows Smart App Control.

### 2. Launch the Application
Double-click `start_tts.bat` (or place a shortcut on your Desktop):
```bat
start_tts.bat
```
The batch script activates the virtual environment, configures the Windows CUDA allocator, launches `app.py`, and automatically opens your browser to:
```
http://localhost:7860
```

---

## 💻 Manual Installation

```bash
# 1. Clone repository
git clone https://github.com/thedreadlorddeath/Voice-Clone.git
cd Voice-Clone

# 2. Create virtual environment (Python 3.11)
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the Studio
python app.py
```

---

## 📖 Usage Guide

1. **Provide a Voice Reference**:
   - Place a clean 5–15s sample named `my_voice.wav` in the project root folder, **or**
   - Upload any `.wav`, `.mp3`, or `.m4a` file in the **Voice Reference Setup** section of the UI.
   - Click **"Denoise & Test Reference"** to verify acoustic filtering.
2. **Input Script**:
   - Paste Markdown text directly into the editor, **or**
   - Click **"Import Obsidian File"** to load any `.md` note.
3. **Inspect Chunks**:
   - Check the live stats bar for character count, word count, chunk count, and estimated audio duration.
   - Expand **"Parsed Emotion Segments"** to review individual chunk styles and parameter assignments.
4. **Generate**:
   - Click **"⚡ Quick Test (First 2 Chunks)"** to verify tone and pacing in seconds.
   - Click **"🎙️ Generate Full Voiceover (.m4a)"** for the complete audio.
5. **Download**:
   - Preview inline with the audio player and click **"⬇️ Download .m4a"** to save your rendered audio.

---

## 🗂️ Project Structure

```
Voice-Clone/
├── app.py               # Main Gradio application, parser, audio cleaner & TTS engine
├── requirements.txt     # Pinned Python package dependencies
├── setup_env.bat        # Automated virtual environment installation script
├── start_tts.bat        # One-click desktop launcher (launches app and opens browser)
├── .gitignore           # Git ignore rules for caches, outputs, and user audio
└── README.md            # Documentation and usage guide
```

---

## 🔧 Technical Details & Troubleshooting

- **Windows Smart App Control**: Specific package versions (`regex==2024.11.6`, `numba==0.61.2`, `scikit-learn==1.5.2`) are pinned to prevent unsigned DLL blocks under Windows 11 Smart App Control.
- **Audio Watermarking Fallback**: Uses `setuptools<70` for `pkg_resources` compatibility with the `perth` audio watermarker, with an automatic fallback to `DummyWatermarker` to prevent runtime crashes.
- **Windows PyTorch Allocator**: Uses `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` to avoid memory fragmentation on 6 GB RTX 3050 laptops without triggering unsupported platform warnings.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
