# PDFQuery AI - Revolutionizing PDFs with Gemini AI

## About the Project
This is a Python application that allows you to load a PDF and ask questions about it using natural language. The application uses a Large Language Model (LLM) to generate responses based on your PDF. The application reads the PDF, splits the text into chunks, and uses Google's text-embedding models to create a vector searchable knowledge base. It then finds the most relevant chunks and feeds them to the Gemini model to generate accurate answers.

## Key Features
 - **PDF Chat**: Interact with your documents using natural language queries.
 - **Gemini Content Generator**: Generate high-quality content using the power of Google's Gemini AI.
 - **English First**: Full support for English interactive interface.

## Required Libraries
 - tiktoken
 - faiss-cpu
 - langchain
 - PyPDF2
 - python-dotenv
 - streamlit
 - google-generativeai

# Installation 

This will guide you on how to set up the project locally.

1. Clone the repo
   ```bash
   git clone https://github.com/KalyanMurapaka45/DocGenius-Revolutionizing-PDFs-with-AI.git
   cd DocGenius-Revolutionizing-PDFs-with-AI
   ```
 
2. Create and activate a virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate # or `.venv\Scripts\activate` on Windows
   ```

3. Install requirements
   ```bash
   pip install -r requirements.txt
   ```

4. Configuration
   Copy `.env.example` to `.env` and add your API key:
   ```bash
   cp .env.example .env
   ```
   Add your `GEMINI_API` key (available from [Google AI Studio](https://aistudio.google.com/app/apikey)).

5. Run the application
   ```bash
   streamlit run app.py
   ```

# Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

# License
Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.

# Acknowledgements
We would like to express our gratitude to the open-source community for their invaluable inspiration and contributions. We also acknowledge the Python libraries used in this project and their respective contributors.
