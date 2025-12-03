import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. SETUP & CLOUD CONFIGURATION
# ==============================================================================
# ### EXPLAINER: This sets up the webpage title and layout.
st.set_page_config(page_title="Leadership Feedback Analyzer", layout="wide")

# --- NLTK DATA HANDLING ---
# ### EXPLAINER: This block is crucial for the app to run on the cloud.
# It downloads the dictionary (tokenizer) needed to split text into sentences.
# We use @st.cache_resource so it only downloads once when the app starts.
@st.cache_resource
def download_nltk_data():
    import nltk
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        # If the standard download fails, we try a backup method often needed for cloud servers.
        st.warning(f"Standard NLTK download failed: {e}. Attempting fallback...")
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception as e_specific:
            st.error(f"Critical Error: Failed to download NLTK data. {e_specific}")
            return False
    return True

if not download_nltk_data():
    st.stop()

# --- UI HEADER ---
st.markdown("<h1 style='text-align:center; font-weight:600;'>Leadership Feedback Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px; color: grey;'>AI-Powered Analysis for Coaching Effectiveness, Clarity & Action Orientation</p>", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD AI MODELS (CACHED)
# ==============================================================================
# ### EXPLAINER: This loads the "Brain" of the AI.
# 1. question_classifier: Decides if feedback is vague or specific.
# 2. sentiment_analyzer: Decides if the tone is positive or negative.
# 3. embedder: Converts sentences into numbers to check if the coach is paraphrasing (Active Listening).
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

with st.spinner("Loading AI Brain... (This happens only once)"):
    question_classifier, sentiment_analyzer, model = load_models()

# ==============================================================================
# 3. SIDEBAR: METRIC CALIBRATION
# ==============================================================================
st.sidebar.header("⚙️ Metric Calibration")
st.sidebar.info("These keywords drive the AI's scoring. Edit them to tune the feedback criteria.")

def get_list_from_string(text_input):
    # ### EXPLAINER: Helper to turn a comma-separated string into a list of words.
    return [x.strip().lower() for x in text_input.split(",") if x.strip()]

# -- CLIENT-APPROVED "GOAL ORIENTED" DEFAULTS --

# 1. Open Questions
# ### EXPLAINER: Words that start good, deep questions.
default_open_starters = "what, how, in what way, to what extent, describe, imagine, what else"
user_open_starters = st.sidebar.text_area("Open Question Starters", default_open_starters, height=70)

# 2. Coaching Keywords
# ### EXPLAINER: Words that show the coach is helping the user discover answers (not advising).
default_coaching = "perspective, insight, awareness, reflect, discover, explore, possibility, challenge, obstacle, outcome, realization"
user_coaching_kws = st.sidebar.text_area("Coaching Keywords (Discovery)", default_coaching, height=70)

# 3. Advisory/Consulting Keywords (PENALTY)
# ### EXPLAINER: This is NEW. These are words we DO NOT want the coach to say (giving advice).
# We will use this list to lower their score if they act like a consultant.
default_advisory = "should, recommend, suggest, advice, i think, you need to, let's work on, i would"
user_advisory_kws = st.sidebar.text_area("Advisory Keywords (Penalty)", default_advisory, height=70, help="If the Coach uses these words, their score will be penalized.")

# 4. Action Keywords
# ### EXPLAINER: Words that show the Coachee is taking ownership (e.g. 'will you' instead of 'we should').
default_action = "will you, by when, deadline, commit, accountability, measure, first step, ownership"
user_action_kws = st.sidebar.text_area("Action/Goal Keywords (Ownership)", default_action, height=70)

# 5. Active Listening
# ### EXPLAINER: Phrases where the coach is validating or clarifying, not interrupting.
default_affirmations = "go on, say more, i hear you, sounds like, what i hear, let me check, am i right"
user_affirmations = st.sidebar.text_area("Active Listening Phrases", default_affirmations, height=70)

# 6. Emotion Keywords
# ### EXPLAINER: Words that show empathy and emotional intelligence.
default_emotions = "feel, frustrated, excited, overwhelmed, confident, worried, energy, sense, connect"
user_emotions = st.sidebar.text_area("Emotion/Empathy Keywords", default_emotions, height=70)

# ==============================================================================
# 4. ANALYSIS FUNCTIONS
# ==============================================================================

# ### EXPLAINER: NEW FUNCTION TO ISOLATE COACH SPEECH
# This function reads the transcript and throws away everything said by the "Coachee".
# It only keeps text that follows the Coach's Label (e.g., "[Speaker 1]").
def extract_coach_speech(transcript, coach_label):
    """
    Parses the transcript and returns ONLY the text spoken by the Coach.
    """
    # Regex to find blocks like: [Speaker 1] (0:00 - 0:05)
    pattern = r"\[(.*?)\]\s*\(\d{1,2}:\d{2}.*?\)"
    
    # Split the text by these speaker blocks
    parts = re.split(pattern, transcript)
    
    coach_text = []
    
    # Loop through the split parts to find the Coach's sections
    for i in range(1, len(parts), 2):
        speaker = parts[i].strip() # The name (e.g., Speaker 1)
        text = parts[i+1].strip()  # What they said
        
        # If the name matches the Coach, keep the text.
        if speaker.lower() == coach_label.lower():
            coach_text.append(text)
            
    # Combine all coach segments into one big text block
    return " ".join(coach_text)

def analyze_transcript(transcript, coach_label):
    if not transcript: return None
    
    # ### EXPLAINER: Step 1 - Isolate Coach Text
    # We call the function above. If the extraction works, we analyze ONLY coach text.
    # If the transcript format is weird and extraction fails (len < 10), we fallback to the whole text safely.
    coach_only_text = extract_coach_speech(transcript, coach_label)
    
    if len(coach_only_text) > 10:
        analysis_text = coach_only_text
    else:
        analysis_text = transcript # Fallback

    # ### EXPLAINER: Step 2 - Clean the text
    # Remove timestamps like (0:02) so they don't mess up word counts.
    clean_text = re.sub(r'\(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\)', '', analysis_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    sentences = sent_tokenize(clean_text)
    
    if len(sentences) == 0: return None

    # Load the keywords from the Sidebar
    open_starters_list = get_list_from_string(user_open_starters)
    coaching_list = get_list_from_string(user_coaching_kws)
    advisory_list = get_list_from_string(user_advisory_kws) # Loaded penalty list
    action_list = get_list_from_string(user_action_kws)
    affirmations_list = get_list_from_string(user_affirmations)
    emotions_list = get_list_from_string(user_emotions)

    # --- METRICS CALCULATIONS ---

    # 1. Question Quality
    # ### EXPLAINER: We check if questions start with "What/How/Imagine" (Good) vs "Do you/Did you" (Closed).
    rule_questions = [s for s in sentences if s.strip().endswith('?')]
    if open_starters_list:
        open_starters_list.sort(key=len, reverse=True) 
        pattern_string = '|'.join([re.escape(x) for x in open_starters_list])
        regex_pattern = r'^(' + pattern_string + r')'
    else:
        regex_pattern = r'^(what|how)'

    def is_open_question(q):
        return re.search(regex_pattern, q.lower().strip()) is not None

    open_questions = [q for q in rule_questions if is_open_question(q)]
    question_score = len(open_questions) / len(rule_questions) if rule_questions else 0

    # 2. Question Stacking
    # ### EXPLAINER: We count if the coach asks multiple questions in one breath (e.g. "Why? Is it bad?").
    question_stacking_count = sum(max(s.count('?') - 1, 0) for s in sentences)

    # 3. Feedback Sentiment (AI)
    # ### EXPLAINER: Use AI to see if feedback words are used in a positive or negative sentence.
    feedback_keywords = ["feedback", "suggest", "recommend", "improve", "consider"]
    feedback_sentences = [s for s in sentences if any(k in s.lower() for k in feedback_keywords)]
    sentiment_results = sentiment_analyzer(feedback_sentences) if feedback_sentences else []
    positive_count = sum(1 for r in sentiment_results if r["label"].lower() == "positive")
    feedback_score = positive_count / max(len(feedback_sentences), 1)

    # 4. Coaching Presence (With Penalty)
    # ### EXPLAINER: First, calculate the "Good" score (Discovery words).
    coaching_statements = [s for s in sentences if any(k in s.lower() for k in coaching_list)]
    coaching_score_raw = len(coaching_statements) / max(len(sentences), 1)

    # ### EXPLAINER: NEW PENALTY LOGIC
    # 1. We count sentences where the Coach gave advice (Advisory keywords).
    # 2. We subtract this from their score.
    advisory_statements = [s for s in sentences if any(k in s.lower() for k in advisory_list)]
    advisory_penalty_ratio = len(advisory_statements) / max(len(sentences), 1)
    
    # We subtract the penalty. If it goes below 0, we keep it at 0.
    coaching_score_final = max(coaching_score_raw - (advisory_penalty_ratio * 1.0), 0)

    # 5. Action Orientation
    # ### EXPLAINER: Check for words that drive commitment ("Will you", "By when").
    action_items = [s for s in sentences if any(k in s.lower() for k in action_list)]
    action_score = len(action_items) / max(len(sentences), 1)

    # 6. Active Listening
    # ### EXPLAINER: Check for validation phrases ("I hear you") AND use AI to check if they paraphrased.
    affirmation_count = sum(1 for s in sentences if any(a in s.lower() for a in affirmations_list))
    paraphrase_count = 0
    if len(sentences) > 1:
        embeddings = model.encode(sentences)
        for i in range(len(sentences) - 1):
            sim = util.cos_sim(embeddings[i], embeddings[i + 1])[0][0]
            if sim > 0.75:
                paraphrase_count += 1
    listening_score = (affirmation_count + paraphrase_count) / max(len(sentences), 1)

    # 7. Language Richness
    # ### EXPLAINER: Check if the coach uses a diverse vocabulary (Unique words / Total words).
    words = [w.lower() for w in clean_text.split() if w.isalpha()]
    language_score = len(set(words)) / max(len(words), 1)

    # 8. Emotion Addressing
    # ### EXPLAINER: Check if the coach uses emotional words (Empathy).
    emotion_sentences = [s for s in sentences if any(k in s.lower() for k in emotions_list)]
    emotion_score = len(emotion_sentences) / max(len(sentences), 1)

    # 9. Feedback Quality (Zero-Shot AI)
    # ### EXPLAINER: Ask the AI if the feedback was "Specific" or "Vague".
    feedback_quality_score = 0
    for s in feedback_sentences:
        result = question_classifier(s, candidate_labels=["vague", "specific", "actionable"])
        if result["labels"][0] in ["specific", "actionable"]:
            feedback_quality_score += 1
    feedback_quality_score = feedback_quality_score / max(len(feedback_sentences), 1) if feedback_sentences else 0

    return {
        "Question Quality": question_score * 100,
        "Question Stacking": (question_stacking_count * 10),
        "Feedback Quality": feedback_quality_score * 100,
        "Active Listening": listening_score * 100,
        "Language Richness": language_score * 100,
        "Emotion Addressing": emotion_score * 100,
        "Coaching Presence": coaching_score_final * 100, # This uses the penalized score
        "Action Orientation": action_score * 100,
        "Positive Feedback": feedback_score * 100
    }

# --- HELPER FUNCTIONS FOR TIME ANALYSIS ---
def parse_time_to_seconds(time_str):
    parts = [int(p) for p in re.findall(r'\d+', time_str)]
    if len(parts) == 2: h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 3: h, m, s = parts
    else: h = m = s = 0
    return h * 3600 + m * 60 + s

def estimate_speaking_time(transcript, coach_label, coachee_label):
    # ### EXPLAINER: This looks at the timestamps to calculate who spoke longer.
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

# ==============================================================================
# 5. UI LAYOUT
# ==============================================================================
st.markdown("### 📂 Upload Transcripts")
col1, col2 = st.columns(2)
before_file = col1.file_uploader("BEFORE Conversation", type=["txt"])
after_file = col2.file_uploader("AFTER Conversation (optional)", type=["txt"])

# ==============================================================================
# 6. DYNAMIC SPEAKER IDENTIFICATION & ANALYSIS
# ==============================================================================

if before_file is not None:
    # Read the file so we can determine speakers
    before_file.seek(0)
    before_text_preview = before_file.read().decode("utf-8")
    
    st.divider()
    st.markdown("### 🎙️ Assign Speaker Roles")
    
    # User selects who is who
    speakers = ["Speaker 1", "Speaker 2"]
    coach_label = st.selectbox("Who is the Coach?", speakers)
    coachee_label = [s for s in speakers if s != coach_label][0]
    
    st.markdown(f"✅ **Coach:** {coach_label}  🤝 **Coachee:** {coachee_label}")

    # Start analysis logic
    st.divider()
    with st.spinner("🔄 Analyzing Transcripts... this may take 30-60 seconds..."):
        try:
            # ### EXPLAINER: We pass 'coach_label' to the function so it filters out the Coachee's words.
            before_scores = analyze_transcript(before_text_preview, coach_label) 
            
            if before_scores is None:
                st.error("The 'Before' transcript appears to be empty or unreadable.")
                st.stop()

            speaking_before = estimate_speaking_time(before_text_preview, coach_label, coachee_label)

            # Analyze 'After' file if it exists
            if after_file:
                after_file.seek(0)
                after_text = after_file.read().decode("utf-8")
                after_scores = analyze_transcript(after_text, coach_label)
                speaking_after = estimate_speaking_time(after_text, coach_label, coachee_label)
            else:
                after_scores = {k: 0 for k in before_scores}
                speaking_after = {"Coach Speaking %": 0, "Coachee Speaking %": 0, "Coach (min)": 0, "Coachee (min)": 0}

            # ==============================================================================
            # 7. DISPLAY RESULTS
            # ==============================================================================
            
            # --- SECTION A: SPEAKING TIME ---
            st.subheader("🗣️ Speaking Time Analysis")
            speak_df = pd.DataFrame({
                "Role": ["Coach", "Coachee"],
                "Before (%)": [speaking_before["Coach Speaking %"], speaking_before["Coachee Speaking %"]],
                "Before (min)": [speaking_before["Coach (min)"], speaking_before["Coachee (min)"]],
                "After (%)": [speaking_after["Coach Speaking %"], speaking_after["Coachee Speaking %"]],
                "After (min)": [speaking_after["Coach (min)"], speaking_after["Coachee (min)"]],
            })
            
            st.dataframe(speak_df.style.format({
                "Before (%)": "{:.1f}", 
                "Before (min)": "{:.1f}", 
                "After (%)": "{:.1f}", 
                "After (min)": "{:.1f}"
            }), use_container_width=True)

            # --- SECTION B: METRICS TABLE ---
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

                # --- SECTION C: VISUALIZATION ---
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

                # --- SECTION D: SUMMARY METRIC ---
                avg_before = np.mean(list(before_scores.values()))
                avg_after = np.mean(list(after_scores.values()))
                change = avg_after - avg_before
                st.subheader("🏆 Overall Effectiveness Change")
                st.metric("Total Improvement", f"{change:+.2f}%")

        except Exception as e:
            st.error(f"Analysis Failed: {e}")
