"""Multimodal Attendance & Engagement Agent
Privacy-first classroom analytics demo. Real identity recognition is intentionally not implemented.
"""
from __future__ import annotations
import io, json, math, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_NAME = "Multimodal Attendance & Engagement Agent"
DATA_DIR = Path("session_data")
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1450px;}
[data-testid="stMetric"] {background: rgba(120,140,170,.08); padding: 14px; border-radius: 12px;}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("Skill Wallet Capstone Project 5 · Privacy-first classroom analytics")
st.info("Privacy demo: processing is designed to run locally. This prototype does not identify people by name or upload media to an external API. Do not use its estimates for grading, discipline, or high-stakes decisions.")

with st.sidebar:
    st.header("Session setup")
    session_name = st.text_input("Session name", "Classroom Session")
    duration_min = st.number_input("Session duration (minutes)", min_value=1, max_value=300, value=50)
    present_threshold = st.slider("Attendance threshold (% of session)", 50, 100, 70)
    st.divider()
    st.caption("For an offline deployment, run this app on the classroom computer and keep recordings on that device. Hosted demos should use synthetic/sample media.")

# Helpers

def make_demo_attendance(n=8, duration_min=50, threshold=70):
    rng = np.random.default_rng(19)
    rows=[]
    for i in range(1, n+1):
        first = int(rng.integers(0, 8))
        last = int(rng.integers(max(first+1, duration_min-12), duration_min+1))
        mins = max(0, last-first)
        pct = mins / duration_min * 100
        rows.append({"Student ID": f"P{i:03d}", "First Seen": f"{first:02d}:00", "Last Seen": f"{last:02d}:00", "Total Duration (min)": mins, "Presence (%)": round(pct,1), "Status": "Present" if pct >= threshold else "Absent"})
    return pd.DataFrame(rows)

def audio_classify(y, sr, frame_seconds=2.0):
    """Lightweight energy-based demo classifier. Not a validated speaker/VAD model."""
    if len(y) == 0: return pd.DataFrame(columns=["Time (s)","RMS Energy","Category"])
    frame=max(1,int(sr*frame_seconds)); rows=[]
    for start in range(0,len(y),frame):
        chunk=y[start:start+frame]
        if len(chunk)==0: continue
        rms=float(np.sqrt(np.mean(np.square(chunk,dtype=np.float64))))
        zcr=float(np.mean(np.abs(np.diff(np.signbit(chunk))))) if len(chunk)>1 else 0
        if rms < 0.012: category="Silence"
        elif rms < 0.055: category="Instructor Lecture / Speech-like"
        elif zcr > 0.17: category="High Ambient Distraction (possible)"
        else: category="Student Discussion / Verbal Interaction (possible)"
        rows.append({"Time (s)":round(start/sr,1),"RMS Energy":round(rms,4),"Category":category})
    return pd.DataFrame(rows)

def engagement_demo(duration_min, audio_df=None):
    rng=np.random.default_rng(42)
    times=np.arange(0,duration_min,1)
    visual=np.clip(72 + 12*np.sin(times/5) + rng.normal(0,7,len(times)),0,100)
    audio=np.full(len(times),55.0)
    if audio_df is not None and not audio_df.empty:
        for i,t in enumerate(times):
            idx=min(len(audio_df)-1, int(t*60/max(1, float(audio_df["Time (s)"].iloc[-1]+1))))
            cat=audio_df["Category"].iloc[idx]
            audio[i]=70 if "Discussion" in cat else 62 if "Lecture" in cat else 38 if "Distraction" in cat else 45
    score=np.clip(0.65*visual+0.35*audio,0,100)
    return pd.DataFrame({"Minute":times,"Visual proxy (demo)":visual.round(1),"Audio interaction proxy":audio.round(1),"Engagement Index":score.round(1)})

# State
if "attendance" not in st.session_state: st.session_state.attendance = None
if "audio_df" not in st.session_state: st.session_state.audio_df = None
if "engagement" not in st.session_state: st.session_state.engagement = None

attendance_tab, audio_tab, fusion_tab, advice_tab, report_tab = st.tabs([
    "1 · Presence & Attendance", "2 · Audio Dynamics", "3 · Fusion & Engagement", "4 · Teaching Advisories", "5 · Session Report"
])

with attendance_tab:
    st.subheader("Presence-based attendance")
    st.write("This runnable demo includes a synthetic register. Video upload is accepted for local experimentation, but reliable multi-person tracking and student-to-ID mapping require calibration and consent; this starter does not claim identity recognition.")
    c1,c2=st.columns([1,1])
    with c1:
        if st.button("Generate synthetic attendance demo", type="primary"):
            st.session_state.attendance=make_demo_attendance(duration_min=duration_min, threshold=present_threshold)
    with c2:
        uploaded_video=st.file_uploader("Optional classroom video (kept in session; not processed in this starter)", type=["mp4","mov","avi"], key="video_upload")
        if uploaded_video: st.warning("Video received by this app session, but no face tracking is run in this starter. For privacy, use only consented footage and do not deploy real classroom footage to a hosted demo.")
    if st.session_state.attendance is not None:
        df=st.session_state.attendance
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download attendance CSV", df.to_csv(index=False).encode("utf-8"), file_name="attendance_register.csv", mime="text/csv")
        present=int((df.Status=="Present").sum())
        st.metric("Attendance rate", f"{present/len(df)*100:.1f}%", f"{present} of {len(df)} demo participants")
    st.caption("Production note: sustained presence should be computed per anonymous track as detected frames / eligible session frames, with occlusion handling and manual review. Face tracking IDs are not student identities.")

with audio_tab:
    st.subheader("Acoustic interaction & noise dynamics")
    st.write("Upload a WAV file for local energy-window analysis. The labels are heuristic proxies, not validated speaker attribution or reliable instructor/student classification.")
    audio_file=st.file_uploader("Upload WAV audio", type=["wav"], key="audio_upload")
    if audio_file:
        try:
            import wave
            with wave.open(io.BytesIO(audio_file.getvalue()), "rb") as wf:
                channels=wf.getnchannels(); width=wf.getsampwidth(); sr=wf.getframerate(); frames=wf.readframes(wf.getnframes())
            if width == 2: y=np.frombuffer(frames,dtype=np.int16).astype(np.float32)/32768.0
            elif width == 4: y=np.frombuffer(frames,dtype=np.int32).astype(np.float32)/2147483648.0
            elif width == 1: y=(np.frombuffer(frames,dtype=np.uint8).astype(np.float32)-128)/128.0
            else: raise ValueError("Unsupported WAV sample width")
            if channels>1: y=y.reshape(-1,channels).mean(axis=1)
            st.audio(audio_file.getvalue(), format="audio/wav")
            if st.button("Analyze audio locally", type="primary"):
                st.session_state.audio_df=audio_classify(y,sr)
        except Exception as e:
            st.error(f"Could not read audio: {e}. Please upload a standard PCM WAV file.")
    if st.session_state.audio_df is not None:
        adf=st.session_state.audio_df
        st.dataframe(adf, use_container_width=True, hide_index=True)
        counts=adf["Category"].value_counts().rename_axis("Category").reset_index(name="Windows")
        st.bar_chart(counts.set_index("Category"))
        st.download_button("⬇ Download audio analysis CSV", adf.to_csv(index=False).encode(), "audio_dynamics.csv", "text/csv")

with fusion_tab:
    st.subheader("Multimodal fusion & engagement timeline")
    st.write("The default visual signal is synthetic. Replace it with validated, consent-based aggregate posture/head-pose features before real classroom use.")
    if st.button("Generate engagement timeline", type="primary"):
        st.session_state.engagement=engagement_demo(duration_min, st.session_state.audio_df)
    if st.session_state.engagement is not None:
        edf=st.session_state.engagement
        avg=float(edf["Engagement Index"].mean()); low=int((edf["Engagement Index"]<45).sum())
        m1,m2,m3=st.columns(3); m1.metric("Mean Engagement Index", f"{avg:.1f}/100"); m2.metric("Minutes below 45", str(low)); m3.metric("Timeline samples", str(len(edf)))
        st.line_chart(edf.set_index("Minute")[["Visual proxy (demo)","Audio interaction proxy","Engagement Index"]])
        st.download_button("⬇ Download engagement timeline", edf.to_csv(index=False).encode(), "engagement_timeline.csv", "text/csv")
        st.caption("Engagement is an experimental aggregate proxy, not a measurement of a student's internal attention or learning.")

with advice_tab:
    st.subheader("Non-punitive teaching suggestion engine")
    if st.session_state.engagement is None:
        st.info("Generate an engagement timeline in Tab 3 to create session-specific advisories.")
    else:
        edf=st.session_state.engagement
        below=edf[edf["Engagement Index"]<45]
        longest=0; run=0
        for val in edf["Engagement Index"]:
            run=run+1 if val<45 else 0; longest=max(longest,run)
        suggestions=[]
        if longest>=5: suggestions.append(("Attention dip",f"Collective engagement proxy stayed below 45 for about {longest} consecutive minutes. Consider a quick poll, think-pair-share, or a 2-minute stretch break."))
        if st.session_state.audio_df is not None and not st.session_state.audio_df.empty:
            dist=(st.session_state.audio_df["Category"].str.contains("Distraction").mean())*100
            if dist>20: suggestions.append(("Possible ambient noise",f"About {dist:.0f}% of audio windows were flagged as high-energy/noise-like. Check room acoustics and ask the class whether instructions are audible."))
        if not suggestions: suggestions.append(("Keep interaction varied","No sustained low-score period was detected in this demo. Continue using brief checks for understanding and varied participation formats."))
        for title,msg in suggestions:
            st.markdown(f"**{title}**")
            st.write(msg)
        st.caption("Advisories are prompts for educator judgment, not automated evaluations of students or teachers.")

with report_tab:
    st.subheader("Session analytics & educator report")
    edf=st.session_state.engagement; adf=st.session_state.attendance; aud=st.session_state.audio_df
    if edf is None and adf is None and aud is None:
        st.info("Run one or more analyses in the other tabs to build a report.")
    else:
        report={"application":APP_NAME,"session_name":session_name,"generated_at":datetime.now().isoformat(timespec="seconds"),"session_duration_minutes":int(duration_min),"note":"Demo analytics only; not for high-stakes decisions."}
        if adf is not None:
            report["attendance"]={"participants_in_register":int(len(adf)),"present":int((adf.Status=="Present").sum()),"attendance_rate_percent":round(float((adf.Status=="Present").mean()*100),1)}
        if aud is not None and not aud.empty:
            report["audio_windows"]={str(k):int(v) for k,v in aud["Category"].value_counts().items()}
        if edf is not None:
            report["engagement"]={"mean_index":round(float(edf["Engagement Index"].mean()),1),"minimum_index":round(float(edf["Engagement Index"].min()),1),"minutes_below_45":int((edf["Engagement Index"]<45).sum())}
        cols=st.columns(3)
        cols[0].metric("Session", session_name)
        cols[1].metric("Duration", f"{duration_min} min")
        cols[2].metric("Report sections", str(len(report)-4))
        st.json(report)
        st.download_button("⬇ Download session report (JSON)", json.dumps(report,indent=2).encode(), "session_report.json", "application/json")
        st.download_button("⬇ Download session report (TXT)", json.dumps(report,indent=2), "session_report.txt", "text/plain")

st.divider()
st.caption("Prototype status: dashboard, synthetic attendance, WAV energy analysis, demo fusion, heuristic advisories, and JSON/CSV exports are implemented. Production face tracking, calibrated VAD, true real-time capture, and validated engagement models remain future work.")
