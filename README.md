# 🎓 AI Video Lecture Summarizer

A Streamlit web app that transforms video lectures into concise, readable summaries using AI. Paste a video URL or drop in a transcript and get a structured summary in seconds — powered by Whisper, TF-IDF, and BART.

🔗 **Live Demo:** https://video-lecture-summarizer-o67mipgcmnjajhutppsa6a.streamlit.app/

---

## What it does

The app runs a 3-step pipeline:

```
Video URL → Audio Extraction → Speech-to-Text → AI Summary
```

1. **Audio extraction** — downloads audio from YouTube, Vimeo, Dailymotion, and 1000+ other platforms using `yt-dlp`
2. **Transcription** — converts speech to text using OpenAI Whisper (via `faster-whisper`), with parallel chunk processing and live progress tracking
3. **Summarisation** — generates a summary using one of two AI models (your choice)

---

## Models

### TF-IDF + Random Forest (Extractive)
- Scores every sentence in the transcript by importance
- Selects and returns the top sentences — nothing is rewritten
- Fast, deterministic, works great for structured lectures
- Shows a confidence score per sentence and a highlighted transcript view

### BART (Abstractive)
- Fine-tuned BART model that reads the transcript and generates new summary text
- Paraphrases and synthesises — does not just copy sentences
- Better for unstructured or conversational lectures
- Handles long transcripts automatically via chunking

---

## Features

- **Video URL input** — supports YouTube, Vimeo, Dailymotion, Facebook, Twitter, and 1000+ more via yt-dlp
- **Manual transcript input** — paste any text directly if you already have a transcript
- **Language selection** — Whisper supports 12 languages with auto-detect as default
- **Session state** — results persist across sidebar changes; no re-transcription on every interaction
- **Re-generate summary** — switch models or adjust length without re-downloading the video
- **Timestamped transcript** — Whisper segment timestamps displayed in a searchable table
- **Highlighted transcript** — sentences colour-coded by TF-IDF confidence (green = high, yellow = medium)
- **Confidence distribution chart** — histogram of sentence importance scores
- **Download as TXT or DOCX** — export the summary and full transcript
- **Dark mode compatible** — all UI elements use CSS variables
- **ffmpeg pre-flight check** — clear install instructions shown if ffmpeg is missing

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Video download | yt-dlp |
| Speech to text | faster-whisper (OpenAI Whisper) |
| Extractive summarisation | TF-IDF + Random Forest (scikit-learn) |
| Abstractive summarisation | Fine-tuned BART (Hugging Face Transformers) |
| Model hosting | Hugging Face Hub |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
video-lecture-summarizer/
├── app.py                      # Main Streamlit application
├── tfidf_lecture_model.pkl     # Trained TF-IDF + Random Forest model
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (ffmpeg)
└── .gitignore
```

> The BART model is hosted on Hugging Face Hub and downloaded automatically at runtime — it is not stored in this repository.

---

## Running Locally

**Prerequisites:**
- Python 3.10+
- ffmpeg installed on your system

**Install ffmpeg:**
```bash
# Linux
sudo apt install ffmpeg

# Mac
brew install ffmpeg

# Windows
winget install ffmpeg
```

**Clone and install:**
```bash
git clone https://github.com/YOUR_USERNAME/video-lecture-summarizer.git
cd video-lecture-summarizer
pip install -r requirements.txt
```

**Run:**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Configuration

All settings are available in the sidebar:

| Setting | Default | Description |
|---|---|---|
| Model | TF-IDF | Choose between extractive (TF-IDF) and abstractive (BART) |
| Summary length | 20% | Target size as a percentage of the original |
| Language | Auto-detect | Whisper transcription language |
| Audio chunk size | 60s | Length of each audio segment for parallel processing |
| Text chunk tokens | 900 | Max tokens per BART chunk |
| Beam count | 4 | BART beam search width (quality vs speed) |
| Length penalty | 2.0 | BART length penalty |
| No-repeat n-gram | 3 | Prevents BART from repeating phrases |
| TF-IDF model path | `tfidf_lecture_model.pkl` | Path to the trained .pkl file |
| BART model path | `YOUR_HF_USERNAME/bart-lecture-model` | Hugging Face repo ID or local path |

---

## RAM & CPU Safety

This app is optimised for consumer hardware (tested on 8 GB RAM / 8-core CPU):

- **Whisper model** auto-selected by video length: `tiny` (< 5 min) · `base` (5–20 min) · `small` (> 20 min) — never loads `medium` or `large` automatically
- **Whisper threads** capped at `min(4, cpu_count // 2)` — always leaves half your cores free
- **BART** runs with `compute_type="int8"` which halves its memory footprint
- **Models** loaded once via `@st.cache_resource` — never reloaded on reruns

---

## Deployment

The app is deployed on **Streamlit Community Cloud**.

Key deployment files:
- `requirements.txt` — all Python packages
- `packages.txt` — contains `ffmpeg` (installed automatically by Streamlit Cloud via apt)

To deploy your own copy:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, set main file to `app.py`, click Deploy

---

## Limitations

- YouTube URLs may fail on cloud deployments due to IP-based rate limiting by YouTube — manual transcript input always works as a fallback
- BART first-run on the live app is slow (~2–3 min) as the model downloads from Hugging Face Hub; subsequent runs are fast
- Very long videos (> 1 hour) may hit Streamlit Cloud's memory limits — use manual transcript input for those

---

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — optimised Whisper inference
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — video/audio downloading
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — BART model
- [Streamlit](https://streamlit.io) — app framework and hosting
