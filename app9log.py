import streamlit as st
import pandas as pd
import re
import nltk
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================
# 1. PAGE SETUP
# ======================================================================
st.set_page_config(page_title="Leadership Feedback Analyzer", layout="wide")

st.markdown("<h1 style='text-align:center; font-weight:600;'>Leadership Feedback Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px; color: grey;'>AI-Powered Analysis for Coaching Effectiveness, Clarity & Action Orientation</p>", unsafe_allow_html=True)

# ======================================================================
# 🚨 FIX: Remove NLTK Downloading — Replace sent_tokenize Safely
# ======================================================================
def safe_sent_tokenize(text):
    """Try NLTK tokenizer, else fallback to regex splitter (no NLTK needed)."""
    try:
        return nltk.sent_tokenize(text)
    except:
        # Simple reliable regex-based fallback
        return re.split(r'(?<=[.!?])\s+', text)

# ======================================================================
# 2. LOAD MODELS
# ======================================================================
@st.cache_resource
def load_models():
    try:
        question_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return question_classifier, sentiment_analyzer, embedder
    except Exception as e:
        st.error(f"Error loading AI models: {e}")
        return None, None, None

with st.spinner("Loading AI Brain... (This only happens once)"):
    question_classifier, sentiment_analyzer, model = load_models()

# ======================================================================
# 3. SIDEBAR CONFIG
# ======================================================================
st.sidebar.header("⚙️ Metric Calibration")
st.sidebar.info("These keywords act as 'Concept Anchors'. The AI looks for semantic matches.")

def get_list_from_string(text_input):
    return [x.strip().lower() for x in text_input.split(",") if x.strip()]

default_open_starters = "what, how, in what way, to what extent, describe, imagine, what else"
user_open_starters = st.sidebar.text_area("Question Quality Keywords", default_open_starters, height=70)

default_coaching = "perspective, insight, awareness, reflect, discover, explore, possibility, challenge, obstacle, outcome, realization"
user_coaching_kws = st.sidebar.text_area("Coaching Presence Keywords", default_coaching, height=70)

default_advisory = "should, recommend, suggest, advice, i think, you need to, let's work on, i would"
user_advisory_kws = st.sidebar.text_area("Advisory Keywords (Penalty)", default_advisory, height=70)

default_action = "will you, by when, deadline, commit, accountability, measure, first step, ownership, next steps"
user_action_kws = st.sidebar.text_area("Action Orientation Keywords", default_action, height=70)

default_affirmations = "go on, say more, i hear you, sounds like, what i hear, let me check, am i right"
user_affirmations = st.sidebar.text_area("Active Listening Keywords", default_affirmations, height=70)

default_emotions = "feel, frustrated, excited, overwhelmed, confident, worried, energy, sense, connect"
user_emotions = st.sidebar.text_area("Emotion Keywords", default_emotions, height=70)

default_feedback_id = "feedback, observation, notice, perspective, reaction, saw, heard, impression"
user_feedback_id_kws = st.sidebar.text_area("Feedback Identification Keywords", default_feedback_id, height=70)

# ======================================================================
# 4. ANALYSIS FUNCTIONS
# ======================================================================
def extract_coach_speech(transcript, coach_label):
    pattern = r"\[(.*?)\]\s*\(\d{1,2}:\d{2}.*?\)"
    parts = re.split(pattern, transcript)
    coach_text = []
    for i in range(1, len(parts), 2):
        speaker = parts[i].strip()
        text = parts[i+1].strip()
        if speaker.lower() == coach_label.lower():
            coach_text.append(text)
    return " ".join(coach_text)

def count_semantic_matches(sentences, keywords, threshold=0.35):
    if not sentences or not keywords:
        return []
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)
    cosine_scores = util.cos_sim(sentence_embeddings, keyword_embeddings)
    max_scores_per_sentence, _ = cosine_scores.max(dim=1)
    matches = max_scores_per_sentence > threshold
    matching_indices = matches.nonzero(as_tuple=True)[0].tolist()
    return [sentences[i] for i in matching_indices]

def analyze_transcript(transcript, coach_label):
    if not transcript:
        return None

    coach_only_text = extract_coach_speech(transcript, coach_label)
    analysis_text = coach_only_text if len(coach_only_text) > 10 else transcript

    clean_text = re.sub(r'\(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\)', '', analysis_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # 🚨 FIX: Replace NLTK sent_tokenize with safe version
    sentences = safe_sent_tokenize(clean_text)

    total_sentences = len(sentences)
    if total_sentences == 0:
        return None

    open_list = get_list_from_string(user_open_starters)
    coaching_list = get_list_from_string(user_coaching_kws)
    action_list = get_list_from_string(user_action_kws)
    affirm_list = get_list_from_string(user_affirmations)
    emotion_list = get_list_from_string(user_emotions)
    feedback_id_list = get_list_from_string(user_feedback_id_kws)
    advisory_list = get_list_from_string(user_advisory_kws)

    # 1. Question Quality
    rule_questions = [s for s in sentences if s.strip().endswith('?')]
    if open_list:
        open_list.sort(key=len, reverse=True)
        pattern_string = '|'.join([re.escape(x) for x in open_list])
        regex_pattern = r'^(' + pattern_string + r')'
    else:
        regex_pattern = r'^(what|how)'

    def is_open_question(q):
        return re.search(regex_pattern, q.lower().strip()) is not None

    open_questions = [q for q in rule_questions if is_open_question(q)]
    question_score = len(open_questions) / len(rule_questions) if rule_questions else 0

    # 2. Question Stacking
    question_stacking = sum(max(s.count('?') - 1, 0) for s in sentences)

    # 3. Feedback Sentiment
    feedback_sents = count_semantic_matches(sentences, feedback_id_list, threshold=0.45)
    sentiment_results = sentiment_analyzer(feedback_sents) if feedback_sents else []
    positive_count = sum(1 for r in sentiment_results if r["label"].lower() == "positive")
    feedback_score = positive_count / max(len(feedback_sents), 1)

    # 4. Coaching Presence
    coaching_sents = count_semantic_matches(sentences, coaching_list, threshold=0.35)
    coaching_score = len(coaching_sents) / max(len(sentences), 1)

    # 5. Action Orientation
    action_sents = count_semantic_matches(sentences, action_list, threshold=0.35)
    action_score = len(action_sents) / max(len(sentences), 1)

    # 6. Active Listening (semantic + paraphrasing)
    affirm_sents = count_semantic_matches(sentences, affirm_list, threshold=0.35)
    affirmation_count = len(affirm_sents)

    paraphrase_count = 0
    if len(sentences) > 1:
        embeddings = model.encode(sentences)
        for i in range(len(sentences) - 1):
            sim = util.cos_sim(embeddings[i], embeddings[i+1])[0][0]
            if sim > 0.75:
                paraphrase_count += 1

    listening_score = (affirmation_count + paraphrase_count) / max(len(sentences), 1)

    # 7. Language Richness
    words = [w.lower() for w in clean_text.split() if w.isalpha()]
    unique_words = set(words)
    language_score = len(unique_words) / max(len(words), 1)

    # 8. Emotion Addressing
    emotion_sents = count_semantic_matches(sentences, emotion_list, threshold=0.35)
    emotion_score = len(emotion_sents) / max(len(sentences), 1)

    # 9. Feedback Quality (Zero-shot)
    specific_count = 0
    for s in feedback_sents:
        result = question_classifier(s, candidate_labels=["vague", "specific", "actionable"])
        if result["labels"][0] in ["specific", "actionable"]:
            specific_count += 1

    feedback_quality_score = specific_count / max(len(feedback_sents), 1)

    return {
        "Question Quality": question_score * 100,
        "Question Stacking": question_stacking * 10,
        "Feedback Quality": feedback_quality_score * 100,
        "Active Listening": listening_score * 100,
        "Language Richness": language_score * 100,
        "Emotion Addressing": emotion_score * 100,
        "Coaching Presence": coaching_score * 100,
        "Action Orientation": action_score * 100,
        "Positive Feedback": feedback_score * 100
    }

# ======================================================================
# (THE REST OF YOUR CODE REMAINS 100% IDENTICAL)
# ======================================================================

# --- TIME FUNCTIONS ---
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

    for speaker, start, end in matches:
        duration = parse_time_to_seconds(end) - parse_time_to_seconds(start)
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

# --- UI ---
st.markdown("### 📂 Upload Transcripts")
col1, col2 = st.columns(2)
before_file = col1.file_uploader("BEFORE Conversation", type=["txt"])
after_file = col2.file_uploader("AFTER Conversation (optional)", type=["txt"])

if before_file:
    before_file.seek(0)
    before_text_preview = before_file.read().decode("utf-8")

    st.divider()
    st.markdown("### 🎙️ Assign Speaker Roles")

    speakers = ["Speaker 1", "Speaker 2"]
    coach_label = st.selectbox("Who is the Coach?", speakers)
    coachee_label = [s for s in speakers if s != coach_label][0]

    st.markdown(f"✅ **Coach:** {coach_label}  🤝 **Coachee:** {coachee_label}")

    st.divider()

    with st.spinner("🔄 Analyzing Transcripts..."):
        try:
            before_scores = analyze_transcript(before_text_preview, coach_label)

            if before_scores is None:
                st.error("The 'Before' transcript appears empty or unreadable.")
                st.stop()

            speaking_before = estimate_speaking_time(before_text_preview, coach_label, coachee_label)

            if after_file:
                after_file.seek(0)
                after_text = after_file.read().decode("utf-8")
                after_scores = analyze_transcript(after_text, coach_label)
                speaking_after = estimate_speaking_time(after_text, coach_label, coachee_label)
            else:
                after_scores = {k: 0 for k in before_scores}
                speaking_after = {
                    "Coach Speaking %": 0,
                    "Coachee Speaking %": 0,
                    "Coach (min)": 0,
                    "Coachee (min)": 0
                }

            # Speaking Time Table
            st.subheader("🗣️ Speaking Time Analysis")
            speak_df = pd.DataFrame({
                "Role": ["Coach", "Coachee"],
                "Before (%)": [speaking_before["Coach Speaking %"], speaking_before["Coachee Speaking %"]],
                "Before (min)": [speaking_before["Coach (min)"], speaking_before["Coachee (min)"]],
                "After (%)": [speaking_after["Coach Speaking %"], speaking_after["Coachee Speaking %"]],
                "After (min)": [speaking_after["Coach (min)"], speaking_after["Coachee (min)"]],
            })

            numeric_cols = ["Before (%)", "Before (min)", "After (%)", "After (min)"]
            st.dataframe(speak_df.style.format({col: "{:.1f}" for col in numeric_cols}),use_container_width=True)


            # Metric Table
            st.subheader("📊 Leadership Communication Score Comparison")

            tips_map = {
                "Question Quality": "Tip: Prioritize 'What' and 'How' questions.",
                "Question Stacking": "Tip: Ask one question at a time.",
                "Feedback Quality": "Tip: Use specific examples.",
                "Active Listening": "Tip: Paraphrase and validate feelings.",
                "Language Richness": "Tip: Vary vocabulary.",
                "Emotion Addressing": "Tip: Name and acknowledge emotions.",
                "Coaching Presence": "Tip: Use discovery questions.",
                "Action Orientation": "Tip: End with commitments.",
                "Positive Feedback": "Tip: Include specific praise."
            }

            metric_names = [
                f"{k}\n({tips_map[k]})" for k in before_scores.keys()
            ]

            df = pd.DataFrame({
                "Metric": metric_names,
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

            st.dataframe(df_display, use_container_width=True)

            # Chart
            st.subheader("📈 Before vs After Comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(df["Metric"]))
            w = 0.35

            ax.bar(x - w/2, df["Before"], w, label="Before", color="#3A7CA5")
            ax.bar(x + w/2, df["After"], w, label="After", color="#7EC480")

            ax.set_xticks(x)
            ax.set_xticklabels(list(before_scores.keys()), rotation=45, ha="right")
            ax.set_ylabel("Score (%)")
            ax.legend()
            st.pyplot(fig)

            # Overall Improvement
            avg_before = np.mean(list(before_scores.values()))
            avg_after = np.mean(list(after_scores.values()))
            change = avg_after - avg_before

            st.subheader("🏆 Overall Effectiveness Change")
            st.metric("Total Improvement", f"{change:+.2f}%")

        except Exception as e:
            st.error(f"Analysis Failed: {e}")

