import os
import time
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader

# Gemini and LangChain imports
import google.generativeai as genai
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

# Configure APIs — supports both local .env and Streamlit Cloud secrets
GEMINI_API = os.getenv("GEMINI_API") or st.secrets.get("GEMINI_API", None)
if GEMINI_API:
    genai.configure(api_key=GEMINI_API)
    os.environ["GOOGLE_API_KEY"] = GEMINI_API


# --- Helper: Retry with exponential backoff ---
def call_with_retry(fn, max_retries=4, initial_delay=15):
    """Call fn(), retrying on quota/rate-limit errors with exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                if attempt < max_retries - 1:
                    st.warning(f"⏳ Rate limit hit. Waiting {delay}s before retrying (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    st.error("❌ Rate limit exceeded after multiple retries. Please wait a minute and try again.")
                    return None
            elif "API_KEY" in err_str or "api key" in err_str.lower() or "invalid" in err_str.lower():
                st.error("🔑 Invalid or missing API key. Please check your GEMINI_API secret in the Streamlit Cloud settings.")
                return None
            else:
                st.error(f"❌ An error occurred: {err_str}")
                return None
    return None


# --- Helper: Load logo ---
def load_app_logo():
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "Artifacts", "Image Resources", "images.jpeg")
    if os.path.exists(IMAGE_PATH):
        return Image.open(IMAGE_PATH)
    return None

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

    def create_embeddings():
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return FAISS.from_texts(chunks, embeddings)

    return call_with_retry(create_embeddings)

# --- Page config ---
app_logo = load_app_logo()
st.set_page_config(
    page_title="PDFQuery AI: Intelligent Document Assistant",
    page_icon=app_logo,
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("PDFQuery AI Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "PDF Chat (PDFQuery AI)", "Gemini Content Generator"])

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "Home":
    st.title("Welcome to PDFQuery AI")
    if app_logo:
        st.image(app_logo, width=200)
    st.write("""
    Welcome to **PDFQuery AI**, the intelligent PDF interaction application powered by Google Gemini.

    With this application, you can:
    - **PDF Chat**: Upload a PDF and ask questions about its content in natural language.
    - **Gemini Generator**: Use Google's Gemini AI to generate content and insights.

    Select a feature from the sidebar to get started.
    """)


# =============================================================================
# PDF CHAT PAGE
# =============================================================================
elif page == "PDF Chat (PDFQuery AI)":
    st.header("Ask Your PDF 📄")

    if not GEMINI_API:
        st.warning("⚠️ GEMINI_API not found. Please set it in your .env file.")
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
                with st.spinner("Processing PDF... (this may take a moment)"):
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
                        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
                        chain = load_qa_chain(llm, chain_type="stuff")
                        return chain.run(input_documents=docs, question=query)

                    response = call_with_retry(get_answer)
                    if response:
                        st.success("Response:")
                        st.write(response)

# =============================================================================
# GEMINI CONTENT GENERATOR PAGE
# =============================================================================
elif page == "Gemini Content Generator":
    st.header("Gemini AI Content Generator ✨")

    if not GEMINI_API:
        st.warning("⚠️ GEMINI_API not found. Please set it in your .env file.")
    else:
        user_input = st.text_area("Enter a prompt:", placeholder="e.g., 'Summarize the key points of machine learning'", height=120)

        if st.button("Generate Content") and user_input:
            with st.spinner("Generating content..."):
                def generate():
                    model = genai.GenerativeModel("gemini-2.0-flash-lite")
                    return model.generate_content(user_input)

                result = call_with_retry(generate)
                if result:
                    st.write("**Response from Gemini AI:**")
                    st.write(result.text)
        elif not user_input:
            st.info("Please enter a prompt to generate content.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by PDFQuery AI Team 🛠️")
