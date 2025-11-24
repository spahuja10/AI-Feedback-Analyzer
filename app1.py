import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Setup
# -------------------------------
nltk.download('punkt', quiet=True)
st.set_page_config(page_title="Leadership Feedback Analyzer", layout="wide")
st.markdown("<h1 style='text-align:center; font-weight:600;'>Leadership Feedback Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px;'>Measure improvement in coaching, clarity & leadership presence.</p>", unsafe_allow_html=True)

# -------------------------------
# Load Models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    question_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return question_classifier, sentiment_analyzer, embedder

question_classifier, sentiment_analyzer, model = load_models()

# -------------------------------
# Transcript Scoring Function
# -------------------------------
def analyze_transcript(transcript):
    clean_text = re.sub(r'\(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\)', '', transcript)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    sentences = sent_tokenize(clean_text)
    if len(sentences) == 0:
        return None

    # Question types
    rule_questions = [s for s in sentences if s.strip().endswith('?')]
    open_questions = [q for q in rule_questions if re.match(r'^(what|how|why|when|which|where)', q.lower())]
    # Question stacking detection
    stacked_penalty = sum(max(s.count('?') - 1, 0) for s in sentences)
    question_score = len(open_questions) / max(len(rule_questions), 1)
    question_score = max(question_score - 0.05 * stacked_penalty, 0)

    # Sentiment (feedback)
    feedback_keywords = ["feedback", "suggest", "recommend", "improve", "consider"]
    feedback_sentences = [s for s in sentences if any(k in s.lower() for k in feedback_keywords)]
    sentiment_results = sentiment_analyzer(feedback_sentences) if feedback_sentences else []
    positive_count = sum(1 for r in sentiment_results if r["label"].lower() == "positive")
    feedback_score = positive_count / max(len(feedback_sentences), 1)

    # Coaching
    coaching_keywords = ["help", "support", "guide", "teach", "assist", "advise"]
    coaching_statements = [s for s in sentences if any(k in s.lower() for k in coaching_keywords)]
    coaching_score = len(coaching_statements) / max(len(sentences), 1)

    # Action orientation
    goal_keywords = ["improve", "develop", "learn", "enhance", "plan", "implement", "follow up", "goal", "action"]
    action_items = [s for s in sentences if any(k in s.lower() for k in goal_keywords)]
    action_score = len(action_items) / max(len(sentences), 1)

    # Active listening
    affirmations = ["I see", "That makes sense", "I understand", "Got it"]
    affirmation_count = sum(1 for s in sentences if any(a.lower() in s.lower() for a in affirmations))
    paraphrase_count = sum(1 for i in range(len(sentences) - 1)
                           if util.cos_sim(model.encode(sentences[i]), model.encode(sentences[i + 1]))[0][0] > 0.75)
    listening_score = (affirmation_count + paraphrase_count) / max(len(sentences), 1)

    # Language richness
    words = [w.lower() for w in clean_text.split() if w.isalpha()]
    unique_words = set(words)
    language_score = len(unique_words) / max(len(words), 1)

    # Emotion addressing
    emotion_keywords = ["feel", "frustrated", "excited", "sad", "comfortable"]
    emotion_sentences = [s for s in sentences if any(k in s.lower() for k in emotion_keywords)]
    emotion_score = len(emotion_sentences) / max(len(sentences), 1)

    # Feedback quality
    feedback_quality_score = 0
    for s in feedback_sentences:
        result = question_classifier(s, candidate_labels=["vague", "specific", "actionable"])
        if result["labels"][0] in ["specific", "actionable"]:
            feedback_quality_score += 1
    feedback_quality_score /= max(len(feedback_sentences), 1)

    return {
        "Question Quality": question_score * 100,
        "Feedback Quality": feedback_quality_score * 100,
        "Active Listening": listening_score * 100,
        "Language Richness": language_score * 100,
        "Emotion Addressing": emotion_score * 100,
        "Coaching Presence": coaching_score * 100,
        "Action Orientation": action_score * 100,
        "Positive Feedback": feedback_score * 100
    }

# -------------------------------
# Speaking Time Estimation
# -------------------------------
def parse_time_to_seconds(time_str):
    parts = [int(p) for p in re.findall(r'\d+', time_str)]
    if len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 3:
        h, m, s = parts
    else:
        h = m = s = 0
    return h * 3600 + m * 60 + s

def estimate_speaking_time(transcript, coach_label, coachee_label):
    pattern = r"\[([^\]]+)\]\s*\((\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\)"
    matches = re.findall(pattern, transcript)
    coach_total = 0
    coachee_total = 0

    for speaker, start_time, end_time in matches:
        duration = parse_time_to_seconds(end_time) - parse_time_to_seconds(start_time)
        if speaker.lower() == coach_label.lower():
            coach_total += duration
        elif speaker.lower() == coachee_label.lower():
            coachee_total += duration

    total = max(coach_total + coachee_total, 1)
    return {
        "Coach Speaking %": (coach_total / total) * 100,
        "Coachee Speaking %": (coachee_total / total) * 100,
        "Coach (min)": coach_total / 60,
        "Coachee (min)": coachee_total / 60
    }

# -------------------------------
# UI File Upload
# -------------------------------
st.markdown("### Upload Transcripts")
col1, col2 = st.columns(2)
before_file = col1.file_uploader("📂 BEFORE Conversation", type=["txt"])
after_file = col2.file_uploader("📂 AFTER Conversation (optional)", type=["txt"])

# -------------------------------
# Select Coach / Coachee
# -------------------------------
st.markdown("### 🎙️ Assign Speaker Roles")
speakers = ["Speaker 1", "Speaker 2"]
coach_label = st.selectbox("Who is the Coach?", speakers)
coachee_label = [s for s in speakers if s != coach_label][0]
st.markdown(f"✅ **Coach:** {coach_label}  🤝 **Coachee:** {coachee_label}")

# -------------------------------
# Processing
# -------------------------------
if before_file:
    before = before_file.read().decode("utf-8")
    before_scores = analyze_transcript(before)
    speaking_before = estimate_speaking_time(before, coach_label, coachee_label)

    if after_file:
        after = after_file.read().decode("utf-8")
        after_scores = analyze_transcript(after)
        speaking_after = estimate_speaking_time(after, coach_label, coachee_label)
    else:
        after_scores = {m: 0 for m in before_scores}
        speaking_after = {"Coach Speaking %": 0, "Coachee Speaking %": 0, "Coach (min)": 0, "Coachee (min)": 0}

    # 🗣️ Speaking Time Display (Before & After)
    st.subheader("🗣️ Speaking Time Analysis")
    speak_df = pd.DataFrame({
        "Role": ["Coach", "Coachee"],
        "Before (%)": [speaking_before["Coach Speaking %"], speaking_before["Coachee Speaking %"]],
        "Before (min)": [speaking_before["Coach (min)"], speaking_before["Coachee (min)"]],
        "After (%)": [speaking_after["Coach Speaking %"], speaking_after["Coachee Speaking %"]],
        "After (min)": [speaking_after["Coach (min)"], speaking_after["Coachee (min)"]],
    })
    st.dataframe(speak_df.style.format({"Before (%)": "{:.1f}", "After (%)": "{:.1f}", "Before (min)": "{:.1f}", "After (min)": "{:.1f}"}))

    # --- Table for Before/After Metrics ---
    df = pd.DataFrame({
        "Metric": before_scores.keys(),
        "Before": before_scores.values(),
        "After": after_scores.values()
    })
    df["Change"] = df["After"] - df["Before"]

    def arrow(v):
        return f"🟢 ↑ {v:.1f}%" if v > 1 else (f"🔴 ↓ {abs(v):.1f}%" if v < -1 else f"➡️ {abs(v):.1f}%")

    df_display = df.copy()
    df_display["Before"] = df_display["Before"].map(lambda x: f"{x:.1f}%")
    df_display["After"] = df_display["After"].map(lambda x: f"{x:.1f}%")
    df_display["Change"] = df["Change"].apply(arrow)

    st.subheader("📊 Leadership Communication Score Comparison")
    st.dataframe(df_display.style.set_properties(**{'text-align': 'center'}), use_container_width=True)

    # --- Comparison Chart ---
    st.subheader("📈 Before vs After Comparison Chart")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df["Metric"]))
    w = 0.35
    ax.bar(x-w/2, df["Before"], w, label="Before", color="#3A7CA5")
    ax.bar(x+w/2, df["After"], w, label="After", color="#7EC480")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Metric"], rotation=45, ha="right")
    ax.set_ylabel("Score (%)")
    ax.legend()
    st.pyplot(fig)

    # --- Overall Leadership Effectiveness ---
    avg_before = np.mean(list(before_scores.values()))
    avg_after = np.mean(list(after_scores.values()))
    change = avg_after - avg_before
    st.subheader("🏆 Overall Leadership Effectiveness Change")
    st.metric("Change in Score", f"{change:+.2f}%")

#Run the below line of code in cmd(command prompt)
#streamlit run app1.py
