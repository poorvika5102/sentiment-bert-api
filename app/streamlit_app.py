"""
streamlit_app.py
A beautiful demo interface for the Sentiment Analysis API.
Run: streamlit run app/streamlit_app.py
"""

import time
import requests
import streamlit as st

# ── Page configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-positive {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .result-negative {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Configuration ──────────────────────────────────────────
API_URL = "http://localhost:8000"   # Change to EC2 IP when deployed


def call_api(text: str) -> dict:
    """Call the prediction API and return result."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the API is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def call_batch_api(texts: list) -> dict:
    """Call the batch prediction API."""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"texts": texts},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="main-title">🎭 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by fine-tuned BERT | 92.4% F1 Score</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Single Analysis", "Batch Analysis", "Model Info"])

# ── Tab 1: Single Analysis ─────────────────────────────────
with tab1:
    st.subheader("Analyze a single text")

    # Example buttons
    st.write("**Try an example:**")
    col1, col2, col3 = st.columns(3)
    
    example_text = ""
    with col1:
        if st.button("😊 Positive example"):
            example_text = "This movie was absolutely fantastic! The acting was superb and the story kept me engaged throughout. Highly recommend to everyone!"
    with col2:
        if st.button("😞 Negative example"):
            example_text = "Complete waste of time. The plot made no sense, the acting was terrible, and I fell asleep twice. Do not watch this."
    with col3:
        if st.button("😐 Neutral example"):
            example_text = "The movie was okay. Some parts were interesting but others felt slow. It's worth watching if you have nothing else to do."

    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        value=example_text,
        height=150,
        placeholder="Type a movie review, tweet, product review, or any text...",
        max_chars=512,
    )

    char_count = len(text_input)
    st.caption(f"{char_count}/512 characters")

    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                start_time = time.time()
                result = call_api(text_input)
                latency = (time.time() - start_time) * 1000

            if result:
                st.divider()

                # Main result
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                pos_score = result["positive_score"]
                neg_score = result["negative_score"]

                if sentiment == "positive":
                    st.success(f"## 😊 POSITIVE SENTIMENT")
                else:
                    st.error(f"## 😞 NEGATIVE SENTIMENT")

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Sentiment",  sentiment.upper())
                col2.metric("Confidence", f"{confidence*100:.1f}%")
                col3.metric("Latency",    f"{latency:.0f}ms")

                # Probability bars
                st.write("**Probability breakdown:**")
                st.write(f"Positive: {pos_score*100:.1f}%")
                st.progress(pos_score)
                st.write(f"Negative: {neg_score*100:.1f}%")
                st.progress(neg_score)
        else:
            st.warning("Please enter some text first.")

# ── Tab 2: Batch Analysis ──────────────────────────────────
with tab2:
    st.subheader("Analyze multiple texts at once")
    st.caption("Enter one text per line (max 32)")

    batch_input = st.text_area(
        "Texts (one per line):",
        height=200,
        placeholder="Great product!\nTerrible service.\nDecent quality for the price.\nLove this so much!",
    )

    if st.button("🔍 Analyze All", type="primary", use_container_width=True):
        texts = [t.strip() for t in batch_input.strip().split("\n") if t.strip()]
        
        if not texts:
            st.warning("Please enter at least one text.")
        elif len(texts) > 32:
            st.error("Maximum 32 texts at once.")
        else:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                result = call_batch_api(texts)

            if result:
                st.success(f"Analyzed {result['total']} texts in {result['processing_time_ms']:.0f}ms")
                
                # Results table
                import pandas as pd
                rows = []
                for r in result["results"]:
                    rows.append({
                        "Text": r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"],
                        "Sentiment": "😊 Positive" if r["sentiment"] == "positive" else "😞 Negative",
                        "Confidence": f"{r['confidence']*100:.1f}%",
                    })
                
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                # Summary
                positives = sum(1 for r in result["results"] if r["sentiment"] == "positive")
                negatives = len(result["results"]) - positives
                
                col1, col2 = st.columns(2)
                col1.metric("😊 Positive", positives)
                col2.metric("😞 Negative", negatives)

# ── Tab 3: Model Info ──────────────────────────────────────
with tab3:
    st.subheader("Model Information")
    
    if st.button("Fetch Model Info"):
        try:
            response = requests.get(f"{API_URL}/model/info", timeout=10)
            info = response.json()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model", info["model_name"])
                st.metric("Max Length", info["max_length"])
            with col2:
                st.metric("Device", info["device"])
                st.metric("Parameters", info.get("parameters", "N/A"))
            
            st.json(info)
        except Exception as e:
            st.error(f"Could not fetch model info: {e}")

    st.divider()
    st.write("**Training Details:**")
    st.write("- Base model: bert-base-uncased (110M parameters)")
    st.write("- Dataset: IMDb Large Movie Review (50K reviews)")
    st.write("- Training: 3 epochs, batch size 16, learning rate 2e-5")
    st.write("- F1 Score (weighted): **92.4%**")
    st.write("- Accuracy: **91.8%**")