# Streamlit App for T5 Text Summarization
# Run this on VS Code after downloading the model from Kaggle

import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

# Page configuration
st.set_page_config(
    page_title="T5 News Summarizer",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 14px;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📰 T5 News Article Summarizer")
st.markdown("Fine-tuned T5 model for abstractive text summarization on CNN/DailyMail dataset")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_length = st.slider("Max Summary Length", 50, 200, 150)
    min_length = st.slider("Min Summary Length", 20, 100, 40)
    num_beams = st.slider("Number of Beams", 2, 8, 4)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This app uses a fine-tuned T5-small model 
    to generate concise summaries of news articles.
    
    **Task:** Encoder-Decoder Model
    **Model:** T5-small
    **Dataset:** CNN/DailyMail
    """)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the fine-tuned T5 model"""
    try:
        import os
        # Check if model exists locally
        model_path = './t5-summarization-finetuned'
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model folder not found at: {model_path}")
            st.info("""
            **Please ensure:**
            1. You've downloaded the model from Kaggle
            2. Extracted it to the same folder as app.py
            3. The folder is named 't5-summarization-finetuned'
            
            Your folder structure should be:
            ```
            your-project/
            ├── app.py
            └── t5-summarization-finetuned/
                ├── config.json
                ├── pytorch_model.bin
                └── ...
            ```
            """)
            st.stop()
        
        # Load model with local_files_only to prevent downloading
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True
        )
        tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Current directory: {os.getcwd()}")
        st.error(f"Files in current directory: {os.listdir('.')}")
        st.stop()

# Load model
with st.spinner("Loading model..."):
    model, tokenizer, device = load_model()
    st.success(f"✅ Model loaded successfully on {device}")

# Function to generate summary
def generate_summary(text, max_len, min_len, beams):
    """Generate summary for input text"""
    # Prepare input
    input_text = "summarize: " + text
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=max_len,
            min_length=min_len,
            num_beams=beams,
            length_penalty=2.0,
            early_stopping=True
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Article")
    
    # Example articles
    example_article = """
    New York (CNN) -- More than 80 Michael Jackson collectibles -- including the late pop star's famous 
    crystal-encrusted white glove -- will be auctioned off in November. The glove was the prized piece of 
    800 items that went up for auction in March 2010, when it sold for $420,000. Profits from the auction 
    will benefit the Shambala Preserve, where Jackson's two Bengal tigers -- Thriller and Sabu -- have been 
    living since 2006. The auction will be held November 20 at The Hard Rock Cafe in Times Square, New York.
    """
    
    if st.button("Load Example Article"):
        st.session_state.input_text = example_article
    
    # Text input
    input_text = st.text_area(
        "Enter or paste your article here:",
        value=st.session_state.get('input_text', ''),
        height=300,
        placeholder="Paste a news article here to generate a summary..."
    )
    
    # Word count
    word_count = len(input_text.split())
    st.caption(f"Word count: {word_count}")
    
    # Generate button
    generate_btn = st.button("🚀 Generate Summary", type="primary", use_container_width=True)

with col2:
    st.subheader("✨ Generated Summary")
    
    if generate_btn:
        if not input_text.strip():
            st.warning("Please enter some text to summarize!")
        else:
            with st.spinner("Generating summary..."):
                start_time = time.time()
                summary = generate_summary(input_text, max_length, min_length, num_beams)
                elapsed_time = time.time() - start_time
            
            # Display summary
            st.markdown(f"""
            <div class="summary-box">
                <h4>Summary:</h4>
                <p>{summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Statistics
            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Original Words", len(input_text.split()))
            with col_b:
                st.metric("Summary Words", len(summary.split()))
            with col_c:
                compression_ratio = (1 - len(summary.split())/len(input_text.split())) * 100
                st.metric("Compression", f"{compression_ratio:.1f}%")
            
            st.caption(f"⏱️ Generated in {elapsed_time:.2f} seconds")
    else:
        st.info("👆 Enter an article and click 'Generate Summary' to see results")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit | Fine-tuned T5 Model | Task 2: Encoder-Decoder Architecture</p>
    </div>
    """, unsafe_allow_html=True)