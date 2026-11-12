from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from voiceguard.audio import load_audio_bytes
from voiceguard.inference import analyze_waveform

DEFAULT_MODEL = Path(
    "runs/wav2vec2_25k_fresh_test/checkpoint.pt"
)

st.set_page_config(
    page_title="VoiceGuard | Audio screening",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --vg-ink: #e8eef8;
        --vg-muted: #9aa9bd;
        --vg-line: rgba(148, 163, 184, .16);
        --vg-panel: #111c2c;
        --vg-panel-soft: #152338;
        --vg-cyan: #62d5db;
        --vg-green: #62d7a3;
        --vg-red: #ff7b83;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(35, 91, 126, .18), transparent 38%),
            #0a1220;
        color: var(--vg-ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: #0e1929;
        border-right: 1px solid var(--vg-line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--vg-muted);
    }
    .vg-brand {
        color: var(--vg-cyan);
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin: 0 0 .55rem;
    }
    .vg-hero {
        padding: 1.65rem 1.8rem;
        margin: .25rem 0 1.6rem;
        border: 1px solid rgba(98, 213, 219, .22);
        border-radius: 20px;
        background: linear-gradient(115deg, rgba(20, 45, 67, .95), rgba(17, 28, 44, .96) 62%, rgba(23, 39, 57, .96));
        box-shadow: 0 18px 55px rgba(0, 0, 0, .18);
    }
    .vg-hero h1 {
        color: #f3f7fc;
        font-size: clamp(2rem, 4vw, 2.8rem);
        line-height: 1.12;
        letter-spacing: -.04em;
        margin: 0 0 .55rem;
    }
    .vg-hero p {
        color: #b7c5d6;
        font-size: 1rem;
        max-width: 760px;
        margin: 0;
    }
    .vg-kicker {
        color: var(--vg-cyan);
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: 0 0 .35rem;
    }
    .vg-card {
        padding: 1.25rem 1.35rem;
        border: 1px solid var(--vg-line);
        border-radius: 16px;
        background: rgba(17, 28, 44, .88);
    }
    .vg-verdict {
        padding: 1.15rem 1.35rem;
        margin: .5rem 0 1rem;
        border: 1px solid var(--vg-line);
        border-left: 5px solid var(--vg-green);
        border-radius: 14px;
        background: var(--vg-panel);
    }
    .vg-verdict.fake { border-left-color: var(--vg-red); }
    .vg-verdict .label {
        color: var(--vg-muted);
        font-size: .74rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .vg-verdict .value {
        color: #f5f8fc;
        font-size: 2rem;
        font-weight: 760;
        line-height: 1.25;
        margin-top: .18rem;
    }
    .vg-note {
        color: var(--vg-muted);
        font-size: .9rem;
        line-height: 1.55;
    }
    div[data-testid="stFileUploader"] {
        padding: .8rem;
        border: 1px dashed rgba(98, 213, 219, .38);
        border-radius: 14px;
        background: rgba(17, 28, 44, .72);
    }
    div[data-testid="stFileUploader"] section {
        border: 0;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        min-height: 3rem;
        border: 0;
        border-radius: 10px;
        color: #07151b;
        font-weight: 750;
        background: linear-gradient(100deg, #70e0db, #8be0b6);
        box-shadow: 0 8px 22px rgba(80, 196, 184, .16);
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        filter: brightness(1.06);
        border: 0;
    }
    div[data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid var(--vg-line);
        border-radius: 14px;
        background: var(--vg-panel);
    }
    div[data-testid="stMetricLabel"] { color: var(--vg-muted); }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--vg-line);
        border-radius: 12px;
        overflow: hidden;
    }
    hr { border-color: var(--vg-line); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="vg-brand">VoiceGuard</div>', unsafe_allow_html=True)
    st.markdown("### Detector settings")
    model_path = Path(
        st.text_input(
            "Model checkpoint",
            str(DEFAULT_MODEL),
            help="Path to a VoiceGuard-compatible trained model checkpoint.",
        )
    )
    model_ready = model_path.is_file()
    if model_ready:
        st.success("Trained checkpoint ready")
    else:
        st.warning("Checkpoint not found")
    st.divider()
    st.markdown("### Benchmark reference")
    st.metric("Held-out accuracy", "87.45%")
    st.metric("Held-out ROC-AUC", "96.21%")
    st.caption(
        "Measured on 2,000 balanced ASVspoof 2021 DF test clips. "
        "These benchmark metrics do not guarantee performance on new audio."
    )
    st.divider()
    st.caption("Local screening only · uploaded audio is not sent to a service.")

st.markdown(
    """
    <div class="vg-hero">
      <div class="vg-kicker">Voice authenticity · screening</div>
      <h1>Check an audio recording</h1>
      <p>Upload a clip to see the model’s real/fake assessment, the score behind it,
      and the time windows that contributed to the result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="vg-kicker">01 · Select audio</div>', unsafe_allow_html=True)
st.subheader("Upload a recording")
st.markdown(
    '<p class="vg-note">Supported formats: WAV, FLAC, and OGG. '
    "For best results, use a clear speech recording.</p>",
    unsafe_allow_html=True,
)
uploaded = st.file_uploader(
    "Audio file",
    type=["wav", "flac", "ogg"],
    help="Audio is analyzed locally with the selected model checkpoint.",
)

audio_bytes = uploaded.getvalue() if uploaded is not None else None
fingerprint = None
if audio_bytes is not None:
    model_stat = model_path.stat() if model_path.is_file() else None
    model_identity = (
        f"{model_path.resolve()}:{model_stat.st_mtime_ns}:{model_stat.st_size}"
        if model_stat
        else str(model_path)
    )
    fingerprint = hashlib.sha256(
        audio_bytes + model_identity.encode("utf-8")
    ).hexdigest()
    st.audio(audio_bytes)

analyze_clicked = st.button(
    "Analyze recording",
    type="primary",
    disabled=uploaded is None or not model_ready,
    use_container_width=True,
)

if not model_ready:
    st.info(
        "Select the trained checkpoint in the sidebar. See README.md for "
        "setup and training instructions."
    )
elif uploaded is None:
    st.markdown(
        '<div class="vg-card vg-note">Your result, explanation, and '
        "time-localized scores will appear here after analysis.</div>",
        unsafe_allow_html=True,
    )

if analyze_clicked and uploaded is not None and fingerprint is not None:
    try:
        waveform, sample_rate = load_audio_bytes(audio_bytes)
        result = analyze_waveform(waveform, sample_rate, model_path)
    except (OSError, ValueError, RuntimeError) as error:
        st.session_state.pop("voiceguard_analysis", None)
        st.error(f"Could not analyze this audio: {error}")
    else:
        st.session_state["voiceguard_analysis"] = {
            "fingerprint": fingerprint,
            "filename": uploaded.name,
            "result": result,
        }

saved_analysis = st.session_state.get("voiceguard_analysis")
if (
    saved_analysis
    and fingerprint is not None
    and saved_analysis["fingerprint"] == fingerprint
):
    result = saved_analysis["result"]
    st.divider()
    st.markdown('<div class="vg-kicker">02 · Screening result</div>', unsafe_allow_html=True)
    st.subheader(saved_analysis["filename"])
    verdict_class = result["prediction"].lower()
    st.markdown(
        f"""
        <div class="vg-verdict {verdict_class}">
          <div class="label">Model assessment</div>
          <div class="value">{result["prediction"]}</div>
          <div class="vg-note">Screening result from this model — not a forensic
          determination of how the recording was created.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    score, threshold, duration, device = st.columns(4)
    score.metric(
        "Mean fake score",
        f'{result["fake_probability"]:.1%}',
        help="Model score; not a calibrated probability that the clip is fake.",
    )
    threshold.metric("Decision threshold", f'{result["threshold"]:.1%}')
    duration.metric("Clip duration", f'{result["duration_seconds"]:.2f} s')
    device.metric("Processing", result["model_device"].upper())

    st.markdown('<div class="vg-kicker">03 · Why this result</div>', unsafe_allow_html=True)
    st.subheader("Decision details")
    for reason in result["reasons"]:
        st.markdown(f"- {reason}")

    st.markdown('<div class="vg-kicker">04 · Locate model evidence</div>', unsafe_allow_html=True)
    st.subheader("Scores across the recording")
    st.markdown(
        f'<p class="vg-note">Each bar is an overlapping audio window. '
        f"Windows at or above the {result['threshold']:.1%} threshold are "
        "flagged. Use the start and end times to review that section.</p>",
        unsafe_allow_html=True,
    )
    region_labels = [
        f'{region["start_seconds"]:.2f}–{region["end_seconds"]:.2f} s'
        for region in result["regions"]
    ]
    st.bar_chart(
        {"Region": region_labels, "Fake score": [
            region["fake_probability"] for region in result["regions"]
        ]},
        x="Region",
        y="Fake score",
        color="#62d5db",
        height=260,
        width="stretch",
    )
    region_rows = [
        {
            "Start (s)": region["start_seconds"],
            "End (s)": region["end_seconds"],
            "Model score": region["fake_probability"],
            "Assessment": "Flagged" if region["flagged"] else "Below threshold",
        }
        for region in result["regions"]
    ]
    st.dataframe(
        region_rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Start (s)": st.column_config.NumberColumn(format="%.2f"),
            "End (s)": st.column_config.NumberColumn(format="%.2f"),
            "Model score": st.column_config.NumberColumn(format="percent"),
        },
    )

    with st.expander("Limitations and interpretation"):
        st.write(result["explanation_limit"])
        st.write(
            "A flagged window indicates elevated model scores, not a confirmed "
            "edited or synthesized segment. The detector was evaluated on a "
            "specific benchmark and may behave differently on other voices, "
            "recording conditions, codecs, or synthesis methods."
        )

    st.download_button(
        "Download result as JSON",
        data=json.dumps(result, indent=2),
        file_name="voiceguard_detection.json",
        mime="application/json",
    )
