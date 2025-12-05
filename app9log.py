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
# Sets up the Streamlit page title and wide layout for better visibility.
st.set_page_config(page_title="Leadership Feedback Analyzer", layout="wide")

# --- NLTK DATA HANDLING (Robust Cloud Fix) ---
# Downloads the 'punkt' tokenizer for sentence splitting, cached to run only once.
@st.cache_resource
def download_nltk_data():
    import nltk
    try:
        # Attempts standard download of the tokenizer.
        nltk.download('punkt', quiet=True)
    except Exception as e:
        # Falls back to downloading 'punkt_tab' if the standard download fails.
        st.warning(f"Standard NLTK download failed: {e}. Attempting fallback...")
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception as e_specific:
            # Stops execution if NLTK data cannot be downloaded.
            st.error(f"Critical Error: Failed to download NLTK data. {e_specific}")
            return False
    return True

# Executes the download function and stops if it fails.
if not download_nltk_data():
    st.stop()

# --- UI HEADER ---
# Displays the main title and subtitle of the application.
st.markdown("<h1 style='text-align:center; font-weight:600;'>Leadership Feedback Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px; color: grey;'>AI-Powered Analysis for Coaching Effectiveness, Clarity & Action Orientation</p>", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD AI MODELS (CACHED)
# ==============================================================================
# Loads AI models for Zero-Shot (Quality), Sentiment (Positive), and Semantic (Similarity) analysis.
@st.cache_resource
def load_models():
    try:
        # Zero-Shot model checks if feedback is Specific, Vague, or Actionable.
        question_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Sentiment model checks if the tone is Positive or Negative.
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        
        #Semantic AI: Sentence Transformer converts text to vectors to find conceptual matches. This sees the keywords and consider all those words for the anlysis which has similar meaning.
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return question_classifier, sentiment_analyzer, embedder
    except Exception as e:
        st.error(f"Error loading AI models: {e}")
        return None, None, None

# Runs the model loading function with a loading spinner.
with st.spinner("Loading AI Brain... (This happens only once)"):
    question_classifier, sentiment_analyzer, model = load_models()

# ==============================================================================
# 3. SIDEBAR: METRIC CALIBRATION
# ==============================================================================
# Configures the sidebar for users to input custom keywords for analysis.
st.sidebar.header("⚙️ Metric Calibration")
st.sidebar.info("These keywords act as 'Concept Anchors'. The AI looks for sentences similar in meaning.")

# Helper function to clean and split comma-separated text inputs into lists.
def get_list_from_string(text_input):
    return [x.strip().lower() for x in text_input.split(",") if x.strip()]

# --- Question Metrics ---
# Allows user to define keywords for identifying Open-Ended Questions.
default_open_starters = "what, how, in what way, to what extent, describe, imagine, what else"
user_open_starters = st.sidebar.text_area("Question Quality Keywords", default_open_starters, height=70)

# --- Coaching Metrics ---
# Allows user to define keywords for identifying Coaching/Discovery moments.
default_coaching = "perspective, insight, awareness, reflect, discover, explore, possibility, challenge, obstacle, outcome, realization"
user_coaching_kws = st.sidebar.text_area("Coaching Presence Keywords", default_coaching, height=70)

# --- Advisory/Penalty Metrics ---
# Allows user to define keywords for advice-giving that should be penalized.This keywords impact the coaching presence metric.
default_advisory = "should, recommend, suggest, advice, i think, you need to, let's work on, i would"
user_advisory_kws = st.sidebar.text_area("Advisory Keywords (Penalty)", default_advisory, height=70, help="If the Coach uses these words, their score will be penalized.")

# --- Action Metrics ---
# Allows user to define keywords for Action Orientation and ownership.
default_action = "will you, by when, deadline, commit, accountability, measure, first step, ownership, next steps"
user_action_kws = st.sidebar.text_area("Action Orientation Keywords", default_action, height=70)

# --- Listening Metrics ---
# Allows user to define keywords for Active Listening and validation.
default_affirmations = "go on, say more, i hear you, sounds like, what i hear, let me check, am i right"
user_affirmations = st.sidebar.text_area("Active Listening Keywords", default_affirmations, height=70)

# --- Emotion Metrics ---
# Allows user to define keywords for identifying Empathy and Emotion addressing.
default_emotions = "feel, frustrated, excited, overwhelmed, confident, worried, energy, sense, connect"
user_emotions = st.sidebar.text_area("Emotion Addressing Keywords", default_emotions, height=70)

# --- Feedback Metrics ---
# Allows user to define keywords to identify feedback sentences for further analysis. These keywords are considered for both Feedback Quality and Positive Feedback Metrics.
default_feedback_id = "feedback, observation, notice, perspective, reaction, saw, heard, impression"
user_feedback_id_kws = st.sidebar.text_area("Feedback Identification Keywords", default_feedback_id, height=70, help="Sentences containing these words will be analyzed for Sentiment and Quality.")

# ==============================================================================
# 4. ANALYSIS FUNCTIONS
# ==============================================================================

# Helper function to parse the transcript and extract only the Coach's text.
def extract_coach_speech(transcript, coach_label):
    """Parses the transcript to get ONLY the text spoken by the Coach."""
    pattern = r"\[(.*?)\]\s*\(\d{1,2}:\d{2}.*?\)"
    parts = re.split(pattern, transcript)
    coach_text = []
    for i in range(1, len(parts), 2):
        speaker = parts[i].strip()
        text = parts[i+1].strip()
        if speaker.lower() == coach_label.lower():
            coach_text.append(text)
    return " ".join(coach_text)

# Helper function to find sentences that semantically match a list of keywords.
def count_semantic_matches(sentences, keywords, threshold=0.35):
    """
    Helper: Finds sentences that semantically match the keywords.
    Returns the list of matching sentences.
    EXPLAINER: This converts sentences and keywords into vectors and calculates cosine similarity.
    If similarity > threshold, it counts as a match.
    """
    if not sentences or not keywords:
        return []
    
    # Converts both sentences and keywords into vector embeddings.
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)
    
    # Calculates similarity scores between all sentences and all keywords.
    cosine_scores = util.cos_sim(sentence_embeddings, keyword_embeddings)
    max_scores_per_sentence, _ = cosine_scores.max(dim=1)
    
    # Identifies matches where similarity exceeds the threshold.
    matches = max_scores_per_sentence > threshold
    matching_indices = matches.nonzero(as_tuple=True)[0].tolist()
    
    # Returns the list of sentences that matched the keywords.
    return [sentences[i] for i in matching_indices]

# Main analysis function to compute all metrics for the transcript.
def analyze_transcript(transcript, coach_label):
    if not transcript: return None
    
    # Isolates the text spoken by the coach; falls back to full text if too short.
    coach_only_text = extract_coach_speech(transcript, coach_label)
    analysis_text = coach_only_text if len(coach_only_text) > 10 else transcript

    # Cleans the text by removing timestamps and extra whitespace.
    clean_text = re.sub(r'\(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\)', '', analysis_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    sentences = sent_tokenize(clean_text)
    total_sentences = len(sentences)
    if total_sentences == 0: return None

    # Loads the keyword lists from the sidebar inputs.
    open_starters_list = get_list_from_string(user_open_starters)
    coaching_list = get_list_from_string(user_coaching_kws)
    action_list = get_list_from_string(user_action_kws)
    affirmations_list = get_list_from_string(user_affirmations)
    emotions_list = get_list_from_string(user_emotions)
    feedback_id_list = get_list_from_string(user_feedback_id_kws)
    advisory_list = get_list_from_string(user_advisory_kws) # Added Advisory List Loading

    # --- METRICS CALCULATIONS (Using Original Formula Logic) ---

    # 1. Question Quality (Ratio of Open / Total Questions)
    # Identifies sentences ending with '?' and checks if they start with Open keywords.
    rule_questions = [s for s in sentences if s.strip().endswith('?')]
    if open_starters_list:
        # Regex is safer for Question Structure (Open vs Closed)
        open_starters_list.sort(key=len, reverse=True) 
        pattern_string = '|'.join([re.escape(x) for x in open_starters_list])
        regex_pattern = r'^(' + pattern_string + r')'
    else:
        regex_pattern = r'^(what|how)'

    # Checks if a question matches the open-ended pattern.
    def is_open_question(q):
        return re.search(regex_pattern, q.lower().strip()) is not None

    # Calculates the ratio of Open Questions to Total Questions.
    open_questions = [q for q in rule_questions if is_open_question(q)]
    
    # FORMULA: Open / Total Questions
    if rule_questions:
        question_score = len(open_questions) / len(rule_questions)
    else:
        question_score = 0

    # 2. Question Stacking (Count)
    # Counts occurrences of multiple '?' in a single sentence block.
    question_stacking_count = sum(max(s.count('?') - 1, 0) for s in sentences)

    # 3. Feedback Sentiment (Hybrid: Semantic Filter -> AI Analysis)
    # Finds sentences related to feedback, then analyzes their sentiment (positive/negative).
    feedback_sentences = count_semantic_matches(sentences, feedback_id_list, threshold=0.45)
    
    sentiment_results = sentiment_analyzer(feedback_sentences) if feedback_sentences else []
    positive_count = sum(1 for r in sentiment_results if r["label"].lower() == "positive")
    
    # FORMULA: Positive / Total Feedback Sentences
    feedback_score = positive_count / max(len(feedback_sentences), 1)

    # 4. Coaching Presence (Semantic)
    # Finds sentences matching coaching/discovery concepts and calculates their density.
    coaching_statements = count_semantic_matches(sentences, coaching_list, threshold=0.35)
    # FORMULA: Coaching Sentences / Total Sentences
    coaching_score = len(coaching_statements) / max(len(sentences), 1)

    # 5. Action Orientation (Semantic)
    # Finds sentences matching action/ownership concepts and calculates their density.
    action_items = count_semantic_matches(sentences, action_list, threshold=0.35)
    # FORMULA: Action Sentences / Total Sentences
    action_score = len(action_items) / max(len(sentences), 1)

    # 6. Active Listening (Semantic + Paraphrasing)
    # Combines semantic matches for affirmations with AI-detected paraphrasing.
    affirmation_sentences = count_semantic_matches(sentences, affirmations_list, threshold=0.35)
    affirmation_count = len(affirmation_sentences)
    
    # Checks for semantic similarity between consecutive sentences to detect paraphrasing.
    paraphrase_count = 0
    if len(sentences) > 1:
        embeddings = model.encode(sentences)
        for i in range(len(sentences) - 1):
            sim = util.cos_sim(embeddings[i], embeddings[i + 1])[0][0]
            if sim > 0.75: paraphrase_count += 1
            
    # FORMULA: (Affirmations + Paraphrases) / Total Sentences
    listening_score = (affirmation_count + paraphrase_count) / max(len(sentences), 1)

    # 7. Language Richness
    # Calculates the ratio of unique words to total words to measure vocabulary diversity.
    words = [w.lower() for w in clean_text.split() if w.isalpha()]
    unique_words = set(words)
    # FORMULA: Unique Words / Total Words
    language_score = len(unique_words) / max(len(words), 1)

    # 8. Emotion Addressing (Semantic)
    # Finds sentences matching emotion/empathy concepts and calculates their density.
    emotion_sentences = count_semantic_matches(sentences, emotions_list, threshold=0.35)
    # FORMULA: Emotion Sentences / Total Sentences
    emotion_score = len(emotion_sentences) / max(len(sentences), 1)

    # 9. Feedback Quality (Hybrid: Semantic Filter -> Zero Shot)
    # Analyzes identified feedback sentences to see if they are specific/actionable using Zero-Shot.
    specific_count = 0
    for s in feedback_sentences:
        result = question_classifier(s, candidate_labels=["vague", "specific", "actionable"])
        if result["labels"][0] in ["specific", "actionable"]:
            specific_count += 1
            
    # FORMULA: Specific Feedback / Total Feedback Sentences
    feedback_quality_score = specific_count / max(len(feedback_sentences), 1)

    # RETURN RAW SCORES SCALED TO 100 (As per original logic)
    return {
        "Question Quality": question_score * 100,
        "Question Stacking": (question_stacking_count * 10),
        "Feedback Quality": feedback_quality_score * 100,
        "Active Listening": listening_score * 100,
        "Language Richness": language_score * 100,
        "Emotion Addressing": emotion_score * 100,
        "Coaching Presence": coaching_score * 100,
        "Action Orientation": action_score * 100,
        "Positive Feedback": feedback_score * 100
    }

# --- HELPER FUNCTIONS FOR TIME ANALYSIS ---
# Parses a time string (e.g., "1:30") into total seconds.
def parse_time_to_seconds(time_str):
    parts = [int(p) for p in re.findall(r'\d+', time_str)]
    if len(parts) == 2: h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 3: h, m, s = parts
    else: h = m = s = 0
    return h * 3600 + m * 60 + s

# Parses transcript timestamps to estimate speaking time for Coach vs Coachee.
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

# ==============================================================================
# 5. UI LAYOUT
# ==============================================================================
# Displays file uploaders for 'Before' and 'After' transcripts.
st.markdown("### 📂 Upload Transcripts")
col1, col2 = st.columns(2)
before_file = col1.file_uploader("BEFORE Conversation", type=["txt"])
after_file = col2.file_uploader("AFTER Conversation (optional)", type=["txt"])

# ==============================================================================
# 6. DYNAMIC SPEAKER IDENTIFICATION & ANALYSIS
# ==============================================================================

# Main execution block: runs when a file is uploaded.
if before_file is not None:
    before_file.seek(0)
    before_text_preview = before_file.read().decode("utf-8")
    
    st.divider()
    st.markdown("### 🎙️ Assign Speaker Roles")
    
    # Allows user to select which speaker is the Coach.
    speakers = ["Speaker 1", "Speaker 2"]
    coach_label = st.selectbox("Who is the Coach?", speakers)
    coachee_label = [s for s in speakers if s != coach_label][0]
    
    st.markdown(f"✅ **Coach:** {coach_label}  🤝 **Coachee:** {coachee_label}")

    st.divider()
    # Runs the analysis with a spinner to indicate progress.
    with st.spinner("🔄 Analyzing Transcripts... this may take 30-60 seconds..."):
        try:
            # Analyze
            before_scores = analyze_transcript(before_text_preview, coach_label) 
            if before_scores is None:
                st.error("The 'Before' transcript appears to be empty or unreadable.")
                st.stop()

            speaking_before = estimate_speaking_time(before_text_preview, coach_label, coachee_label)

            # Analyses the 'After' transcript if provided.
            if after_file:
                after_file.seek(0)
                after_text = after_file.read().decode("utf-8")
                after_scores = analyze_transcript(after_text, coach_label)
                speaking_after = estimate_speaking_time(after_text, coach_label, coachee_label)
            else:
                # Sets default scores if no 'After' file is provided.
                after_scores = {k: 0 for k in before_scores}
                speaking_after = {"Coach Speaking %": 0, "Coachee Speaking %": 0, "Coach (min)": 0, "Coachee (min)": 0}

            # Results
            # Displays speaking time analysis table.
            st.subheader("🗣️ Speaking Time Analysis")
            speak_df = pd.DataFrame({
                "Role": ["Coach", "Coachee"],
                "Before (%)": [speaking_before["Coach Speaking %"], speaking_before["Coachee Speaking %"]],
                "Before (min)": [speaking_before["Coach (min)"], speaking_before["Coachee (min)"]],
                "After (%)": [speaking_after["Coach Speaking %"], speaking_after["Coachee Speaking %"]],
                "After (min)": [speaking_after["Coach (min)"], speaking_after["Coachee (min)"]],
            })
            
            format_dict = {
                "Before (%)": "{:.1f}", 
                "Before (min)": "{:.1f}", 
                "After (%)": "{:.1f}", 
                "After (min)": "{:.1f}"
            }
            st.dataframe(speak_df.style.format(format_dict), use_container_width=True)

            st.subheader("📊 Leadership Communication Score Comparison")
            if before_scores:
                # --- IMPROVEMENT TIPS LOGIC ---
                # Maps each metric to a specific improvement tip.
                tips_map = {
                    "Question Quality": "Tip: Prioritize 'What' and 'How' questions over closed-ended ones.",
                    "Question Stacking": "Tip: Ask one question at a time and pause for the answer.",
                    "Feedback Quality": "Tip: Use specific examples rather than general praise or criticism.",
                    "Active Listening": "Tip: Use validation phrases like 'I hear you' and paraphrase key points.",
                    "Language Richness": "Tip: Vary your vocabulary to keep engagement high.",
                    "Emotion Addressing": "Tip: Explicitly acknowledge feelings (e.g., 'It sounds frustrating').",
                    "Coaching Presence": "Tip: Use discovery questions to help them find answers. Avoid advice.",
                    "Action Orientation": "Tip: End with clear commitment questions: 'What will you do by when?'",
                    "Positive Feedback": "Tip: Ensure constructive feedback is balanced with encouragement."
                }

                metric_display_names = [f"{k} \n({tips_map[k]})" for k in before_scores.keys()]

                # Creates and displays the main metrics comparison table.
                df = pd.DataFrame({
                    "Metric": metric_display_names,
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
                
                st.dataframe(df_display.style.set_properties(**{'text-align': 'center', 'white-space': 'pre-wrap'}), use_container_width=True)

                # Displays the comparison bar chart.
                st.subheader("📈 Before vs After Comparison")
                fig, ax = plt.subplots(figsize=(8, 4))
                x = np.arange(len(df["Metric"]))
                w = 0.35
                ax.bar(x - w/2, df["Before"], w, label="Before", color="#3A7CA5")
                ax.bar(x + w/2, df["After"], w, label="After", color="#7EC480")
                ax.set_xticks(x)
                ax.set_xticklabels(before_scores.keys(), rotation=45, ha="right")
                ax.set_ylabel("Score (%)")
                ax.legend()
                st.pyplot(fig)

                # Calculates and displays the overall improvement metric.
                avg_before = np.mean(list(before_scores.values()))
                avg_after = np.mean(list(after_scores.values()))
                change = avg_after - avg_before
                st.subheader("🏆 Overall Effectiveness Change")
                st.metric("Total Improvement", f"{change:+.2f}%")

        except Exception as e:
            st.error(f"Analysis Failed: {e}")
