# app.py
import streamlit as st
import pandas as pd
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Version B — Fixed: NO NLTK, regex sentence splitter (Option 1)
# ==============================================================================

# 1. Streamlit page config
st.set_page_config(page_title="Leadership Feedback Analyzer", layout="wide")
st.markdown("<h1 style='text-align:center; font-weight:600;'>Leadership Feedback Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px; color: grey;'>AI-Powered Analysis for Coaching Effectiveness, Clarity & Action Orientation</p>", unsafe_allow_html=True)

# 2. Model loading (cached)
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

with st.spinner("Loading AI models (this runs once)..."):
    question_classifier, sentiment_analyzer, model = load_models()

if any(m is None for m in (question_classifier, sentiment_analyzer, model)):
    st.error("One or more models failed to load. Please check logs and internet access.")
    st.stop()

# 3. Sidebar - configuration / keywords
st.sidebar.header("⚙️ Metric Calibration (edit to tune)")
def get_list_from_string(text_input):
    return [x.strip().lower() for x in text_input.split(",") if x.strip()]

default_open_starters = "what, how, in what way, to what extent, describe, imagine, what else, tell me about"
user_open_starters = st.sidebar.text_area("Open Question Starters", default_open_starters, height=90)

default_coaching = "perspective, insight, awareness, reflect, discover, explore, possibility, challenge, obstacle, outcome, realization"
user_coaching_kws = st.sidebar.text_area("Coaching Keywords (Discovery)", default_coaching, height=90)

default_advisory = "should, recommend, suggest, advice, i think, you need to, let's work on, i would"
user_advisory_kws = st.sidebar.text_area("Advisory Keywords (Penalty)", default_advisory, height=90)

default_action = "will you, by when, deadline, commit, accountability, measure, first step, ownership, next steps"
user_action_kws = st.sidebar.text_area("Action/Goal Keywords (Ownership)", default_action, height=90)

default_affirmations = "go on, say more, i hear you, sounds like, what i hear, let me check, am i right"
user_affirmations = st.sidebar.text_area("Active Listening Phrases", default_affirmations, height=90)

default_emotions = "feel, frustrated, excited, overwhelmed, confident, worried, energy, sense, connect"
user_emotions = st.sidebar.text_area("Emotion/Empathy Keywords", default_emotions, height=90)

default_feedback_id = "feedback, observation, notice, perspective, reaction, saw, heard, impression"
user_feedback_id_kws = st.sidebar.text_area("Feedback Identification Keywords", default_feedback_id, height=90)

# 4. Helpers: regex sentence splitter (Option 1)
def split_sentences_regex(text):
    """
    Simple, robust sentence splitter using regex:
    splits on end-of-sentence punctuation . ? ! followed by whitespace.
    Keeps punctuation with sentence.
    """
    if not text:
        return []
    text = text.strip()
    # Normalize newlines and multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Split on punctuation boundaries
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    # Remove empty parts
    return [p.strip() for p in parts if p.strip()]

# 5. Helpers: safe encoding (tries tensor, falls back to numpy)
def safe_encode(texts, convert_to_tensor=True):
    try:
        return model.encode(texts, convert_to_tensor=convert_to_tensor)
    except Exception:
        # fallback without tensor
        return model.encode(texts, convert_to_tensor=False)

# 6. Helper: extract coach text from transcript blocks like [Speaker 1] (0:02 - 1:45) <text>
def extract_coach_speech(transcript, coach_label):
    pattern = r"\[([^\]]+)\]\s*\(\s*\d{1,2}:\d{2}[^)]*\)\s*(.*?)\s*(?=(?:\[[^\]]+\]\s*\(\s*\d{1,2}:\d{2})|$)"
    # Using findall with DOTALL to capture multi-line segments
    matches = re.findall(pattern, transcript, flags=re.DOTALL)
    coach_text_segments = []
    for speaker, text in matches:
        if speaker.strip().lower() == coach_label.strip().lower():
            coach_text_segments.append(text.strip())
    return " ".join(coach_text_segments)

# 7. Helper: semantic matching using sentence transformer
def count_semantic_matches(sentences, keywords, threshold=0.35):
    """
    Return list of sentences matching ANY keyword semantically (cosine sim > threshold).
    If keywords empty -> returns []
    """
    if not sentences or not keywords:
        return []
    # encode
    s_embeddings = safe_encode(sentences, convert_to_tensor=True)
    k_embeddings = safe_encode(keywords, convert_to_tensor=True)
    try:
        cosine_scores = util.cos_sim(s_embeddings, k_embeddings)
        max_scores_per_sentence, _ = cosine_scores.max(dim=1)
        # Depending on tensor vs np, convert to list of floats
        try:
            bool_mask = (max_scores_per_sentence > threshold).cpu().numpy().tolist()
        except Exception:
            bool_mask = (max_scores_per_sentence > threshold).tolist()
        matching_indices = [i for i, flag in enumerate(bool_mask) if flag]
        return [sentences[i] for i in matching_indices]
    except Exception:
        # On any failure fallback to keyword substring match (safe)
        lower_keywords = [k.lower() for k in keywords]
        return [s for s in sentences if any(k in s.lower() for k in lower_keywords)]

# 8. Core analysis function (updated as requested)
def analyze_transcript(transcript, coach_label):
    if not transcript:
        return None

    # isolate coach text if present
    coach_only_text = extract_coach_speech(transcript, coach_label)
    analysis_text = coach_only_text if len(coach_only_text) > 10 else transcript

    # remove timestamps from the analysis text for metrics (we still keep original for speaker-time)
    clean_text = re.sub(r'\(\s*\d{1,2}:\d{2}[^)]*\)', '', analysis_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    sentences = split_sentences_regex(clean_text)
    total_sentences = len(sentences)
    if total_sentences == 0:
        return None

    # lists from sidebar
    open_starters = get_list_from_string(user_open_starters)
    coaching_kws = get_list_from_string(user_coaching_kws)
    advisory_kws = get_list_from_string(user_advisory_kws)
    action_kws = get_list_from_string(user_action_kws)
    affirmations_kws = get_list_from_string(user_affirmations)
    emotion_kws = get_list_from_string(user_emotions)
    feedback_id_kws = get_list_from_string(user_feedback_id_kws)

    # 1) Question Quality (semantic + pattern)
    rule_questions = [s for s in sentences if s.strip().endswith('?')]
    if open_starters:
        open_starters.sort(key=len, reverse=True)
        pattern_string = '|'.join([re.escape(x) for x in open_starters])
        regex_pattern = r'^(' + pattern_string + r')'
    else:
        regex_pattern = r'^(what|how|why|when|which|where|tell me about|describe)'
    def is_open_question(q):
        return re.search(regex_pattern, q.lower().strip()) is not None
    open_questions = [q for q in rule_questions if is_open_question(q)]
    question_score = (len(open_questions) / len(rule_questions)) if rule_questions else 0

    # 2) Question Stacking
    question_stacking_count = sum(max(s.count('?') - 1, 0) for s in sentences)

    # 3) Positive Feedback (semantic + sentiment)
    feedback_sentences = count_semantic_matches(sentences, feedback_id_kws, threshold=0.45)
    sentiment_results = sentiment_analyzer(feedback_sentences) if feedback_sentences else []
    positive_count = sum(1 for r in sentiment_results if r.get("label","").lower()=="positive")
    positive_feedback_score = positive_count / max(len(feedback_sentences), 1)

    # 4) Coaching Presence (semantic) + advisory penalty
    coaching_matches = count_semantic_matches(sentences, coaching_kws, threshold=0.35)
    coaching_score_raw = len(coaching_matches) / max(len(sentences), 1)
    advisory_matches = count_semantic_matches(sentences, advisory_kws, threshold=0.35)
    advisory_penalty_ratio = len(advisory_matches) / max(len(sentences), 1)
    coaching_score = max(coaching_score_raw - advisory_penalty_ratio, 0)

    # 5) Action Orientation (semantic)
    action_matches = count_semantic_matches(sentences, action_kws, threshold=0.35)
    action_score = len(action_matches) / max(len(sentences), 1)

    # 6) Active Listening (semantic affirmations + paraphrase detection)
    affirmation_matches = count_semantic_matches(sentences, affirmations_kws, threshold=0.35)
    affirmation_count = len(affirmation_matches)
    paraphrase_count = 0
    if len(sentences) > 1:
        try:
            embeddings = safe_encode(sentences, convert_to_tensor=True)
            for i in range(len(sentences)-1):
                sim = util.cos_sim(embeddings[i], embeddings[i+1])[0][0]
                # If tensor result, convert to float for comparison
                try:
                    sim_val = float(sim.cpu().numpy())
                except Exception:
                    sim_val = float(sim)
                if sim_val > 0.75:
                    paraphrase_count += 1
        except Exception:
            # fallback: no paraphrase detection
            paraphrase_count = 0
    listening_score = (affirmation_count + paraphrase_count) / max(len(sentences), 1)

    # 7) Language richness
    words = [w.lower() for w in clean_text.split() if w.isalpha()]
    language_score = len(set(words)) / max(len(words), 1)

    # 8) Emotion addressing (semantic)
    emotion_matches = count_semantic_matches(sentences, emotion_kws, threshold=0.35)
    emotion_score = len(emotion_matches) / max(len(sentences), 1)

    # 9) Feedback Quality (semantic filter -> zero-shot classification)
    specific_count = 0
    for s in feedback_sentences:
        try:
            result = question_classifier(s, candidate_labels=["vague","specific","actionable"])
            top_label = result.get("labels", [None])[0]
            if top_label in ["specific","actionable"]:
                specific_count += 1
        except Exception:
            # fallback: if classifier fails, attempt heuristic by checking length / presence of action words
            if any(a in s.lower() for a in action_kws) or len(s.split())>8:
                specific_count += 1
    feedback_quality_score = specific_count / max(len(feedback_sentences), 1) if feedback_sentences else 0

    return {
        "Question Quality": question_score * 100,
        "Question Stacking": (question_stacking_count * 10),   # scaled for display
        "Feedback Quality": feedback_quality_score * 100,
        "Active Listening": listening_score * 100,
        "Language Richness": language_score * 100,
        "Emotion Addressing": emotion_score * 100,
        "Coaching Presence": coaching_score * 100,
        "Action Orientation": action_score * 100,
        "Positive Feedback": positive_feedback_score * 100
    }

# 9. Time parsing helpers (unchanged)
def parse_time_to_seconds(time_str):
    parts = [int(p) for p in re.findall(r'\d+', time_str)]
    if len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 3:
        h, m, s = parts
    else:
        h = m = s = 0
    return h*3600 + m*60 + s

def estimate_speaking_time(transcript, coach_label, coachee_label):
    pattern = r"\[([^\]]+)\]\s*\((\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\)"
    matches = re.findall(pattern, transcript)
    coach_total = 0
    coachee_total = 0
    for speaker, start_time, end_time in matches:
        duration = parse_time_to_seconds(end_time) - parse_time_to_seconds(start_time)
        if speaker.strip().lower() == coach_label.strip().lower():
            coach_total += max(duration, 0)
        elif speaker.strip().lower() == coachee_label.strip().lower():
            coachee_total += max(duration, 0)
    total = max(coach_total + coachee_total, 1)
    return {
        "Coach Speaking %": (coach_total/total)*100,
        "Coachee Speaking %": (coachee_total/total)*100,
        "Coach (min)": coach_total/60,
        "Coachee (min)": coachee_total/60
    }

# 10. UI: Upload & run
st.markdown("### 📂 Upload Transcripts")
col1, col2 = st.columns(2)
before_file = col1.file_uploader("BEFORE Conversation (.txt)", type=["txt"])
after_file = col2.file_uploader("AFTER Conversation (optional, .txt)", type=["txt"])

if before_file:
    before_file.seek(0)
    before_text = before_file.read().decode("utf-8")

    st.divider()
    st.markdown("### 🎙️ Assign Speaker Roles")
    speakers = ["Speaker 1", "Speaker 2"]
    coach_label = st.selectbox("Who is the Coach?", speakers)
    coachee_label = [s for s in speakers if s != coach_label][0]
    st.markdown(f"✅ **Coach:** {coach_label}  🤝 **Coachee:** {coachee_label}")

    st.divider()
    with st.spinner("🔄 Analyzing transcripts (30–60s first run)..."):
        try:
            before_scores = analyze_transcript(before_text, coach_label)
            if before_scores is None:
                st.error("Before transcript produced no analyzable content.")
                st.stop()
            speaking_before = estimate_speaking_time(before_text, coach_label, coachee_label)

            if after_file:
                after_file.seek(0)
                after_text = after_file.read().decode("utf-8")
                after_scores = analyze_transcript(after_text, coach_label)
                speaking_after = estimate_speaking_time(after_text, coach_label, coachee_label)
            else:
                after_scores = {k: 0 for k in before_scores}
                speaking_after = {"Coach Speaking %": 0, "Coachee Speaking %": 0, "Coach (min)": 0, "Coachee (min)": 0}

            # Display speaking time first (Before & After)
            st.subheader("🗣️ Speaking Time Analysis")
            speak_df = pd.DataFrame({
                "Role": ["Coach", "Coachee"],
                "Before (%)": [speaking_before["Coach Speaking %"], speaking_before["Coachee Speaking %"]],
                "Before (min)": [speaking_before["Coach (min)"], speaking_before["Coachee (min)"]],
                "After (%)": [speaking_after["Coach Speaking %"], speaking_after["Coachee Speaking %"]],
                "After (min)": [speaking_after["Coach (min)"], speaking_after["Coachee (min)"]],
            })
            st.dataframe(speak_df.style.format({"Before (%)":"{:.1f}", "Before (min)":"{:.1f}", "After (%)":"{:.1f}", "After (min)":"{:.1f}"}), use_container_width=True)

            # Tips map (improvement hints shown next to metric labels)
            tips_map = {
                "Question Quality": "(Improve: ask clear openers like 'what/how')",
                "Question Stacking": "(Improve: ask one question, pause for answer)",
                "Feedback Quality": "(Improve: make feedback specific & actionable)",
                "Active Listening": "(Improve: paraphrase & validate feelings)",
                "Language Richness": "(Improve: vary vocabulary & avoid repetition)",
                "Emotion Addressing": "(Improve: name feelings & empathize explicitly)",
                "Coaching Presence": "(Improve: use discovery questions, avoid advising)",
                "Action Orientation": "(Improve: end with commitment & deadlines)",
                "Positive Feedback": "(Improve: balance praise with specific examples)"
            }

            metric_display_names = [f"{k} {tips_map.get(k,'')}" for k in before_scores.keys()]

            # Metrics table
            st.subheader("📊 Leadership Communication Score Comparison")
            df = pd.DataFrame({
                "Metric": metric_display_names,
                "Before": before_scores.values(),
                "After": [after_scores.get(k, 0) for k in before_scores.keys()]
            })
            df["Change"] = df["After"] - df["Before"]

            def arrow(v):
                return f"🟢 +{v:.1f}%" if v > 1 else (f"🔴 {v:.1f}%" if v < -1 else f"➡️ {v:.1f}%")

            df_display = df.copy()
            df_display["Before"] = df_display["Before"].map(lambda x: f"{x:.1f}%")
            df_display["After"] = df_display["After"].map(lambda x: f"{x:.1f}%")
            df_display["Change"] = df["Change"].apply(arrow)

            st.dataframe(df_display.style.set_properties(**{'text-align': 'center', 'white-space': 'pre-wrap'}), use_container_width=True)

            # Comparison chart (compact)
            st.subheader("📈 Before vs After Comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(df["Metric"]))
            w = 0.35
            ax.bar(x - w/2, df["Before"].astype(float), w, label="Before", color="#3A7CA5")
            ax.bar(x + w/2, df["After"].astype(float), w, label="After", color="#7EC480")
            ax.set_xticks(x)
            ax.set_xticklabels(list(before_scores.keys()), rotation=45, ha="right")
            ax.set_ylabel("Score (%)")
            ax.legend()
            st.pyplot(fig)

            # Summary metric
            avg_before = np.mean(list(before_scores.values()))
            avg_after = np.mean(list(after_scores.values()))
            change = avg_after - avg_before
            st.subheader("🏆 Overall Effectiveness Change")
            st.metric("Total Improvement", f"{change:+.2f}%")

        except Exception as e:
            st.error(f"Analysis Failed: {e}")

