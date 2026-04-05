import os
import time
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader

# Groq and LangChain imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configure API key — supports both local .env and Streamlit Cloud secrets
GROQ_API_KEY = None
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
except (KeyError, FileNotFoundError):
    pass

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# --- Helper: Retry with exponential backoff ---
def call_with_retry(fn, max_retries=3, initial_delay=5):
    """Call fn(), retrying on quota/rate-limit errors with exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower():
                if attempt < max_retries - 1:
                    st.warning(f"⏳ Rate limit hit. Waiting {delay}s before retrying (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    st.error("❌ Rate limit exceeded after multiple retries. Please wait a minute and try again.")
                    return None
            elif "401" in err_str or "authentication_error" in err_str.lower() or "invalid_api_key" in err_str.lower():
                st.error("🔑 Invalid API key. Please check your GROQ_API_KEY.")
                return None
            else:
                st.error(f"❌ Error: {err_str}")
                return None
    return None


# --- Helper: Load logo ---
def load_app_logo():
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "Artifacts", "Image Resources", "images.jpeg")
    if os.path.exists(IMAGE_PATH):
        return Image.open(IMAGE_PATH)
    return None


# --- Helper: Get embeddings model (cached to avoid reloading) ---
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    """Load HuggingFace embeddings model locally — no API key needed, no rate limits."""
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- Helper: Build knowledge base from PDF (cached in session_state) ---
def build_knowledge_base(pdf_file):
    """Read PDF and build FAISS vector store. Returns the knowledge_base or None on failure."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if not text.strip():
        st.error("Could not extract text from this PDF. Please try a different file.")
        return None

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = get_embeddings_model()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base


# --- Helper: Get Groq LLM ---
def get_groq_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.3):
    """Create a Groq-backed LLM instance."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=GROQ_API_KEY,
    )


# --- Page config ---
app_logo = load_app_logo()
st.set_page_config(
    page_title="PDFQuery AI: Intelligent Document Assistant",
    page_icon=app_logo,
    layout="wide"
)

# --- Hide entire toolbar (Deploy button + hamburger menu) ---
st.markdown("""
    <style>
        [data-testid="stToolbar"] {display: none !important;}
        #MainMenu {display: none !important;}
        header {visibility: hidden !important; height: 0 !important;}
        footer {visibility: hidden !important;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("PDFQuery AI Navigation")

# Dark / Light mode toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # Default to dark

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode

# Apply theme via CSS
if dark_mode:
    st.markdown("""
        <style>
            /* Main area */
            [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #ffffff; }
            [data-testid="stHeader"] { background-color: #0e1117; }

            /* Sidebar */
            [data-testid="stSidebar"] { background-color: #161b22; }
            [data-testid="stSidebar"] * { color: #ffffff !important; }
            [data-testid="stSidebar"] .stMarkdown p { color: #ffffff !important; }
            [data-testid="stSidebar"] label { color: #ffffff !important; }
            [data-testid="stSidebar"] .stSelectbox label { color: #ffffff !important; }

            /* All text elements */
            h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
            p, span, label, div { color: #e0e0e0; }
            .stMarkdown, .stMarkdown p { color: #e0e0e0 !important; }

            /* Inputs */
            .stTextInput > div > div > input { background-color: #1e2530; color: #ffffff !important; border: 1px solid #3a3f4b; }
            .stTextArea > div > div > textarea { background-color: #1e2530; color: #ffffff !important; border: 1px solid #3a3f4b; }
            .stSelectbox > div > div { background-color: #1e2530; color: #ffffff !important; }
            .stSelectbox > div > div > div { color: #ffffff !important; }

            /* Buttons */
            .stButton > button { background-color: #6C63FF; color: #ffffff !important; border: none; }
            .stButton > button:hover { background-color: #5a52e0; color: #ffffff !important; }

            /* File uploader */
            [data-testid="stFileUploader"] { color: #ffffff !important; }
            [data-testid="stFileUploader"] label { color: #ffffff !important; }
            [data-testid="stFileUploader"] small { color: #cccccc !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #1a1a1a; }
            [data-testid="stSidebar"] { background-color: #f0f2f6; }
            [data-testid="stHeader"] { background-color: #ffffff; }
            h1, h2, h3, h4, h5, h6 { color: #1a1a1a !important; }
            .stTextInput > div > div > input { background-color: #ffffff; color: #1a1a1a; }
            .stTextArea > div > div > textarea { background-color: #ffffff; color: #1a1a1a; }
            .stButton > button { background-color: #6C63FF; color: #ffffff !important; border: none; }
            .stButton > button:hover { background-color: #5a52e0; }
        </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
page = st.sidebar.selectbox("Go to", ["Home", "PDF Chat (PDFQuery AI)", "AI Content Generator"])

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "Home":
    st.title("Welcome to PDFQuery AI")
    if app_logo:
        st.image(app_logo, width=200)
    st.write("""
    Welcome to **PDFQuery AI**, the intelligent PDF interaction application.

    With this application, you can:
    - **PDF Chat**: Upload a PDF and ask questions about its content in natural language.
    - **AI Content Generator**: Generate content and insights using AI.

    Select a feature from the sidebar to get started.
    """)


# =============================================================================
# PDF CHAT PAGE
# =============================================================================
elif page == "PDF Chat (PDFQuery AI)":
    st.header("Ask Your PDF 📄")

    if not GROQ_API_KEY:
        st.warning("⚠️ GROQ_API_KEY not found. Please set it in your .env file or Streamlit Cloud secrets.")
    else:
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

        # Initialize session state keys
        if "knowledge_base" not in st.session_state:
            st.session_state.knowledge_base = None
        if "last_pdf_name" not in st.session_state:
            st.session_state.last_pdf_name = None

        # Only rebuild the knowledge base when a new PDF is uploaded
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.last_pdf_name:
                with st.spinner("📖 Reading your PDF..."):
                    kb = build_knowledge_base(uploaded_file)
                    if kb is not None:
                        st.session_state.knowledge_base = kb
                        st.session_state.last_pdf_name = uploaded_file.name
                        st.success(f"✅ '{uploaded_file.name}' processed successfully! You can now ask questions.")
            else:
                st.info(f"📄 Using cached knowledge base for **{uploaded_file.name}**. Ready for your questions!")

        # Show query input only if we have a knowledge base
        if st.session_state.knowledge_base is not None:
            query = st.text_input("Ask your question about the PDF:")
            if st.button("Get Answer") and query:
                with st.spinner("Analyzing your question..."):
                    def get_answer():
                        docs = st.session_state.knowledge_base.similarity_search(query)
                        llm = get_groq_llm()
                        chain = load_qa_chain(llm, chain_type="stuff")
                        return chain.run(input_documents=docs, question=query)

                    response = call_with_retry(get_answer)
                    if response:
                        st.success("Response:")
                        st.write(response)

# =============================================================================
# AI CONTENT GENERATOR PAGE
# =============================================================================
elif page == "AI Content Generator":
    st.header("AI Content Generator ✨")

    if not GROQ_API_KEY:
        st.warning("⚠️ GROQ_API_KEY not found. Please set it in your .env file or Streamlit Cloud secrets.")
    else:
        user_input = st.text_area("Enter a prompt:", placeholder="e.g., 'Summarize the key points of machine learning'", height=120)

        if st.button("Generate Content") and user_input:
            with st.spinner("Generating content..."):
                def generate():
                    llm = get_groq_llm(model="llama-3.1-8b-instant", temperature=0.7)
                    response = llm.invoke(user_input)
                    return response.content

                result = call_with_retry(generate)
                if result:
                    st.write("**Response from Groq AI:**")
                    st.write(result)
        elif not user_input:
            st.info("Please enter a prompt to generate content.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by Anshika** 💜")
