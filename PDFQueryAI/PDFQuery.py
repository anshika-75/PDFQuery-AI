from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

load_dotenv()
from PIL import Image

# Load assets
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "Artifacts", "Image Resources", "images.jpeg")
if os.path.exists(IMAGE_PATH):
    img = Image.open(IMAGE_PATH)
else:
    img = None

st.set_page_config(page_title="PDFQuery AI: Intelligent Document Assistant", page_icon=img)

st.header("Ask Your PDF📄")
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    query = st.text_input("Ask your Question about your PDF")
    if query:
        docs = knowledge_base.similarity_search(query)

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        st.success(response)