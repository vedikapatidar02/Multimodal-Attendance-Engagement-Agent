# Multimodal Attendance & Engagement Agent

A privacy-aware Streamlit capstone prototype for aggregate classroom analytics.

## Current capabilities
- Five-tab Streamlit dashboard
- Synthetic presence-based attendance register with configurable threshold and CSV export
- Local PCM WAV energy-window analysis with heuristic audio categories and CSV export
- Synthetic visual engagement proxy plus audio-informed timeline
- Non-punitive educator advisories
- Downloadable JSON/TXT session report

## Important limitations
This is a **prototype**, not a validated surveillance or attendance product. It does not implement identity recognition, reliable multi-person face tracking, head-pose estimation, posture classification, real-time microphone capture, speaker diarization, or validated voice activity detection. Audio categories are heuristic and cannot reliably distinguish instructor speech, student speech, and noise. Engagement values are synthetic/experimental proxies and must not be used for grading, discipline, or individual evaluation.

## Run locally
Python 3.9+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For true offline operation, install dependencies before disconnecting from the internet and run on the classroom computer. Keep any real recordings local and obtain informed consent.

## Streamlit Community Cloud demo
1. Push `app.py`, `requirements.txt`, and this README to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
3. Copy the generated public app URL into your submission form.
4. Use synthetic data or non-sensitive sample audio in the hosted demo. A hosted app is not itself 100% offline; local-only processing and public cloud hosting are different deployment modes.

## Suggested GitHub repository description
Privacy-first classroom analytics prototype using Streamlit, Python, audio energy analysis, synthetic presence tracking, engagement timeline visualization, and educator-focused session reports.
