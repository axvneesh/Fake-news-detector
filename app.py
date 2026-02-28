import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from pathlib import Path

# Optional: `newspaper3k` is useful for URL scraping but may not be installed in all environments.
try:
    from newspaper import Article
    HAS_NEWSPAPER = True
except Exception:
    Article = None
    HAS_NEWSPAPER = False

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VerifyNews AI",
    page_icon="🔍",
    layout="centered"
)

# --- 2. BACKEND ENGINE (Data & ML) ---

def _train_from_dfs(df_fake: pd.DataFrame, df_true: pd.DataFrame):
    """Train the TF-IDF vectorizer and classifier from two dataframes."""
    def find_text_column(df):
        candidates = ['text', 'article', 'content', 'body', 'article_body', 'headline', 'summary']
        for c in candidates:
            if c in df.columns:
                return c
        for c in df.columns:
            if df[c].dtype == object:
                return c
        return None

    col_fake = find_text_column(df_fake)
    col_true = find_text_column(df_true)
    if col_fake is None or col_true is None:
        st.error("Could not locate a text-like column in one of the CSVs. Please ensure a 'text' column exists.")
        return None, None

    df_fake = df_fake.rename(columns={col_fake: 'text'})
    df_true = df_true.rename(columns={col_true: 'text'})

    df_fake['label'] = 'FAKE'
    df_true['label'] = 'REAL'

    df = pd.concat([df_fake, df_true], axis=0)
    df = df[df['text'].notna() & df['text'].str.strip().ne('')]
    if df.empty:
        st.error("No usable text data found after cleaning the datasets.")
        return None, None

    if len(df) > 15000:
        df = df.sample(frac=1, random_state=42).head(15000)
    else:
        df = df.sample(frac=1, random_state=42)

    tfidf_v = TfidfVectorizer(stop_words='english', max_df=0.8)
    x = tfidf_v.fit_transform(df['text'])
    y = df['label']

    clf = PassiveAggressiveClassifier(max_iter=50)
    clf.fit(x, y)
    return tfidf_v, clf


# FIX 1: Use @st.cache_resource so training only runs once, not on every rerender.
@st.cache_resource
def train_model():
    """Try to load CSVs from disk and train. Returns (tfidf_v, model) or (None, None)."""
    try:
        p_fake = Path("Fake.csv")
        p_true = Path("True.csv")
        if p_fake.exists() and p_true.exists():
            df_fake = pd.read_csv(p_fake)
            df_true = pd.read_csv(p_true)
            return _train_from_dfs(df_fake, df_true)
        else:
            return None, None
    except Exception as e:
        st.error(f"Failed to train model from CSVs: {e}")
        return None, None


# FIX 2: Clean session_state initialization — just call once and store results.
if 'tfidf_v' not in st.session_state or 'model' not in st.session_state:
    tfidf_v, model = train_model()
    st.session_state['tfidf_v'] = tfidf_v
    st.session_state['model'] = model
    if model is None:
        st.warning("Datasets (Fake.csv / True.csv) not found in project folder. You can upload them using the uploader below.")


# --- 3. FRONTEND UI ---
st.title("🕵️ VerifyNews AI Detector")
st.markdown("---")

# Offer uploaders if model not available
if st.session_state.get('model') is None:
    with st.expander("Upload datasets and train the model 🔧", expanded=True):
        fake_file = st.file_uploader("Upload Fake.csv", type=["csv"], key="fake_upload")
        true_file = st.file_uploader("Upload True.csv", type=["csv"], key="true_upload")
        if st.button("Train with uploaded files"):
            if fake_file and true_file:
                try:
                    df_fake = pd.read_csv(fake_file)
                    df_true = pd.read_csv(true_file)
                    with st.spinner("Training model from uploaded files..."):
                        tfidf_v_new, model_new = _train_from_dfs(df_fake, df_true)
                        if model_new is not None:
                            st.session_state['tfidf_v'] = tfidf_v_new
                            st.session_state['model'] = model_new
                            st.success("✅ Model trained successfully.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Failed to read/train from uploaded files: {e}")
            else:
                st.warning("Please upload both Fake.csv and True.csv to train the model.")

if st.session_state.get("model") is not None:
    st.info("⚠️ Results are probabilistic and based on linguistic patterns — always verify news through trusted sources.", icon="ℹ️")

    tab1, tab2 = st.tabs(["🌐 Analyze via Link", "📝 Analyze via Text"])

    # TAB 1: URL SCRAPING
    with tab1:
        # FIX 3: Guard the button logic so it only runs if newspaper3k is available.
        if not HAS_NEWSPAPER:
            st.error("URL scraping requires the `newspaper3k` package. Install it with `pip install newspaper3k` to enable URL analysis.")
        else:
            url_input = st.text_input("Paste News URL:", placeholder="https://example.com/news-article")
            if st.button("Check Link"):
                if url_input:
                    try:
                        with st.spinner("🔍 AI is reading the website content..."):
                            article = Article(url_input)
                            article.download()
                            article.parse()

                            st.info(f"**Found Article:** {article.title}")

                            prediction = st.session_state["model"].predict(
                                st.session_state["tfidf_v"].transform([article.text])
                            )[0]

                            if prediction == 'REAL':
                                st.success("✅ **Result: Likely Credible News**")
                                st.balloons()
                            else:
                                st.error("🚨 **Result: Potential Misinformation/Fake**")
                    except Exception as e:
                        st.warning(f"Could not scrape the URL. It might be blocked by the website. Error: {e}")
                else:
                    st.warning("Please enter a URL first.")

    # TAB 2: MANUAL TEXT PASTE
    # FIX 4: Removed redundant inner model check — already guaranteed by outer `if` block.
    with tab2:
        text_input = st.text_area("Paste Article Text:", height=250, placeholder="Copy and paste the article body here...")
        if st.button("Check Text"):
            if text_input.strip():
                with st.spinner("Analyzing linguistic patterns..."):
                    prediction = st.session_state["model"].predict(
                        st.session_state["tfidf_v"].transform([text_input])
                    )[0]

                    if prediction == 'REAL':
                        st.success("✅ **Result: This text follows credible news structures.**")
                    else:
                        st.error("🚨 **Result: This text contains markers of fake news.**")
            else:
                st.warning("Please paste some text to analyze.")

# --- 4. FOOTER ---
st.markdown("---")
st.caption("Developed as a CS Engineering Prototype | Powered by NLP & Machine Learning")
