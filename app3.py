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
# 2. Load Models (Cached)
# -------------------------------
@st.cache_resource
def load_models():
    question_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return question_classifier, sentiment_analyzer, embedder

question_classifier, sentiment_analyzer, model = load_models()

# -------------------------------
# 3. Sidebar: Metric Calibration
# -------------------------------
st.sidebar.header("⚙️ Metric Calibration")
st.sidebar.info("Modify these lists to auto-update the scoring logic.")

def get_list_from_string(text_input):
    return [x.strip().lower() for x in text_input.split(",") if x.strip()]

# -- Sidebar Inputs --
default_open_starters = "what, how, why, when, which, where, tell me about, describe"
user_open_starters = st.sidebar.text_area("Open Question Starters", default_open_starters, height=70)

default_coaching = "help, support, guide, teach, assist, advise, mentor, strategy"
user_coaching_kws = st.sidebar.text_area("Coaching Keywords", default_coaching, height=70)

default_action = "improve, develop, learn, enhance, plan, implement, follow up, goal, action, next steps"
user_action_kws = st.sidebar.text_area("Action/Goal Keywords", default_action, height=70)

default_affirmations = "i see, that makes sense, i understand, got it, i hear you, right, okay"
user_affirmations = st.sidebar.text_area("Active Listening Phrases", default_affirmations, height=70)

default_emotions = "feel, frustrated, excited, sad, comfortable, worried, happy, angry, nervous"
user_emotions = st.sidebar.text_area("Emotion/Empathy Keywords", default_emotions, height=70)

# -------------------------------
# 4. Analysis Functions
# -------------------------------
def analyze_transcript(transcript):
    if not transcript: return None
    
    clean_text = re.sub(r'\(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\)', '', transcript)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    sentences = sent_tokenize(clean_text)
    if len(sentences) == 0: return None

    open_starters_list = get_list_from_string(user_open_starters)
    coaching_list = get_list_from_string(user_coaching_kws)
    action_list = get_list_from_string(user_action_kws)
    affirmations_list = get_list_from_string(user_affirmations)
    emotions_list = get_list_from_string(user_emotions)

    # 1. Question types
    rule_questions = [s for s in sentences if s.strip().endswith('?')]
    open_questions = [q for q in rule_questions if any(q.lower().startswith(starter) for starter in open_starters_list)]
    
    stacked_penalty = sum(max(s.count('?') - 1, 0) for s in sentences)
    question_score = len(open_questions) / max(len(rule_questions), 1)
    question_score = max(question_score - 0.05 * stacked_penalty, 0)

    # 2. Sentiment (Feedback)
    feedback_keywords = ["feedback", "suggest", "recommend", "improve", "consider"]
    feedback_sentences = [s for s in sentences if any(k in s.lower() for k in feedback_keywords)]
    sentiment_results = sentiment_analyzer(feedback_sentences) if feedback_sentences else []
    positive_count = sum(1 for r in sentiment_results if r["label"].lower() == "positive")
    feedback_score = positive_count / max(len(feedback_sentences), 1)

    # 3. Coaching Presence
    coaching_statements = [s for s in sentences if any(k in s.lower() for k in coaching_list)]
    coaching_score = len(coaching_statements) / max(len(sentences), 1)

    # 4. Action Orientation
    action_items = [s for s in sentences if any(k in s.lower() for k in action_list)]
    action_score = len(action_items) / max(len(sentences), 1)

    # 5. Active Listening
    affirmation_count = sum(1 for s in sentences if any(a in s.lower() for a in affirmations_list))
    paraphrase_count = 0
    if len(sentences) > 1:
        embeddings = model.encode(sentences)
        for i in range(len(sentences) - 1):
            sim = util.cos_sim(embeddings[i], embeddings[i + 1])[0][0]
            if sim > 0.75:
                paraphrase_count += 1
    listening_score = (affirmation_count + paraphrase_count) / max(len(sentences), 1)

    # 6. Language Richness
    words = [w.lower() for w in clean_text.split() if w.isalpha()]
    unique_words = set(words)
    language_score = len(unique_words) / max(len(words), 1)

    # 7. Emotion Addressing
    emotion_sentences = [s for s in sentences if any(k in s.lower() for k in emotions_list)]
    emotion_score = len(emotion_sentences) / max(len(sentences), 1)

    # 8. Feedback Quality
    feedback_quality_score = 0
    for s in feedback_sentences:
        result = question_classifier(s, candidate_labels=["vague", "specific", "actionable"])
        if result["labels"][0] in ["specific", "actionable"]:
            feedback_quality_score += 1
    feedback_quality_score = feedback_quality_score / max(len(feedback_sentences), 1) if feedback_sentences else 0

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

def parse_time_to_seconds(time_str):
    parts = [int(p) for p in re.findall(r'\d+', time_str)]
    if len(parts) == 2: h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 3: h, m, s = parts
    else: h = m = s = 0
    return h * 3600 + m * 60 + s

def estimate_speaking_time(transcript, coach_label, coachee_label):
    pattern = r"\[([^\]]+)\]\s*\((\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\)"
    matches = re.findall(pattern, transcript)
    coach_total = 0
    coachee_total = 0
    for speaker, start_time, end_time in matches:
        duration = parse_time_to_seconds(end_time) - parse_time_to_seconds(start_time)
        if speaker.lower() == coach_label.lower(): coach_total += duration
        elif speaker.lower() == coachee_label.lower(): coachee_total += duration
    total = max(coach_total + coachee_total, 1)
    return {
        "Coach Speaking %": (coach_total / total) * 100,
        "Coachee Speaking %": (coachee_total / total) * 100,
        "Coach (min)": coach_total / 60,
        "Coachee (min)": coachee_total / 60
    }

# -------------------------------
# 5. UI Layout
# -------------------------------
st.markdown("### 📂 Upload Transcripts")
col1, col2 = st.columns(2)
before_file = col1.file_uploader("BEFORE Conversation", type=["txt"])
after_file = col2.file_uploader("AFTER Conversation (optional)", type=["txt"])

st.markdown("### 🎙️ Assign Speaker Roles")
speakers = ["Speaker 1", "Speaker 2"]
coach_label = st.selectbox("Who is the Coach?", speakers)
coachee_label = [s for s in speakers if s != coach_label][0]
st.markdown(f"✅ **Coach:** {coach_label}  🤝 **Coachee:** {coachee_label}")

# -------------------------------
# 6. AUTO-RUN Analysis Logic
# -------------------------------

if before_file is not None:
    st.divider()
    with st.spinner("🔄 Analyzing Transcripts... this may take 30-60 seconds..."):
        try:
            # IMPORTANT: Reset file pointer
            before_file.seek(0)
            before_text = before_file.read().decode("utf-8")
            
            # Analyze Before
            before_scores = analyze_transcript(before_text)
            
            # Safety Check: If analysis returns None (empty transcript), stop here
            if before_scores is None:
                st.error("The 'Before' transcript appears to be empty or unreadable.")
                st.stop()

            speaking_before = estimate_speaking_time(before_text, coach_label, coachee_label)

            # Analyze After (if exists)
            if after_file:
                after_file.seek(0)
                after_text = after_file.read().decode("utf-8")
                after_scores = analyze_transcript(after_text)
                speaking_after = estimate_speaking_time(after_text, coach_label, coachee_label)
            else:
                after_scores = {k: 0 for k in before_scores}
                speaking_after = {"Coach Speaking %": 0, "Coachee Speaking %": 0, "Coach (min)": 0, "Coachee (min)": 0}

            # -------------------------------
            # 7. Display Results
            # -------------------------------
            
            # --- Speaking Time ---
            st.subheader("🗣️ Speaking Time Analysis")
            speak_df = pd.DataFrame({
                "Role": ["Coach", "Coachee"],
                "Before (%)": [speaking_before["Coach Speaking %"], speaking_before["Coachee Speaking %"]],
                "Before (min)": [speaking_before["Coach (min)"], speaking_before["Coachee (min)"]],
                "After (%)": [speaking_after["Coach Speaking %"], speaking_after["Coachee Speaking %"]],
                "After (min)": [speaking_after["Coach (min)"], speaking_after["Coachee (min)"]],
            })
            
            # FIX: Only format the numeric columns, not the 'Role' string column
            st.dataframe(speak_df.style.format({
                "Before (%)": "{:.1f}", 
                "Before (min)": "{:.1f}", 
                "After (%)": "{:.1f}", 
                "After (min)": "{:.1f}"
            }), use_container_width=True)

            # --- Metrics Table ---
            st.subheader("📊 Leadership Communication Score Comparison")
            if before_scores:
                df = pd.DataFrame({
                    "Metric": before_scores.keys(),
                    "Before": before_scores.values(),
                    "After": after_scores.values()
                })
                df["Change"] = df["After"] - df["Before"]

                def arrow(v):
                    return f"🟢 +{v:.1f}%" if v > 1 else (f"🔴 {v:.1f}%" if v < -1 else f"➡️ {v:.1f}%")

                df_display = df.copy()
                df_display["Before"] = df_display["Before"].map(lambda x: f"{x:.1f}%")
                df_display["After"] = df_display["After"].map(lambda x: f"{x:.1f}%")
                df_display["Change"] = df["Change"].apply(arrow)
                
                st.dataframe(df_display.style.set_properties(**{'text-align': 'center'}), use_container_width=True)

                # --- Chart ---
                st.subheader("📈 Before vs After Comparison")
                fig, ax = plt.subplots(figsize=(8, 4))
                x = np.arange(len(df["Metric"]))
                w = 0.35
                ax.bar(x - w/2, df["Before"], w, label="Before", color="#3A7CA5")
                ax.bar(x + w/2, df["After"], w, label="After", color="#7EC480")
                ax.set_xticks(x)
                ax.set_xticklabels(df["Metric"], rotation=45, ha="right")
                ax.set_ylabel("Score (%)")
                ax.legend()
                st.pyplot(fig)

                # --- Overall Score ---
                avg_before = np.mean(list(before_scores.values()))
                avg_after = np.mean(list(after_scores.values()))
                change = avg_after - avg_before
                st.subheader("🏆 Overall Effectiveness Change")
                st.metric("Total Improvement", f"{change:+.2f}%")

        except Exception as e:
            st.error(f"Analysis Failed: {e}")
