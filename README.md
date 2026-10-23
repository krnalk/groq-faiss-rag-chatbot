# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit. It reads PDF files from a local folder, indexes them in a FAISS vector store, and answers questions using only the retrieved document context.

## Features

- Loads all PDFs from a `data/` folder
- Splits documents into overlapping chunks (1000 characters, 200 overlap)
- Creates embeddings with OpenAI and stores them in FAISS
- Answers questions with a Groq-hosted LLM (`openai/gpt-oss-120b`)
- Restricts answers to the provided context to reduce hallucination
- Shows the retrieved source chunks in an expandable section
- Simple Streamlit interface with a loading spinner

## Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| UI               | Streamlit                           |
| LLM              | Groq (`openai/gpt-oss-120b`)        |
| Embeddings       | OpenAI Embeddings                   |
| Vector store     | FAISS                               |
| PDF loading      | LangChain `PyPDFDirectoryLoader`    |
| Text splitting   | `RecursiveCharacterTextSplitter`    |
| Orchestration    | LangChain retrieval chain           |

## How It Works

1. **Load:** PDFs in `data/` are read page by page.
2. **Split:** Pages are cut into overlapping chunks.
3. **Embed and store:** Each chunk is embedded with OpenAI and saved in an in-memory FAISS index.
4. **Retrieve:** Your question is embedded and the most similar chunks are fetched.
5. **Generate:** The chunks and your question are sent to the Groq LLM, which answers from that context only.

## Project Structure

```
.
├── app.py            # Streamlit application
├── data/             # Put your PDF files here
├── .env              # API keys (not committed)
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)
- An [OpenAI API key](https://platform.openai.com/) (used for embeddings)

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt**

```
streamlit
langchain
langchain-classic
langchain-community
langchain-core
langchain-groq
langchain-openai
langchain-text-splitters
faiss-cpu
pypdf
python-dotenv
```

## Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

Make sure `.env` is listed in `.gitignore`.

## Usage

1. Add your PDF files to the `data/` folder.
2. Start the app:
   ```bash
   streamlit run app.py
   ```
3. Click **Generate embeddings** and wait for "Vector Database is Ready!".
4. Type your question and click **Answer**.
5. Open **Context from the documents** to see which chunks were used.

## Notes and Limitations

- You must click **Generate embeddings** before asking a question.
- The FAISS index lives in session memory, so it is rebuilt each time the app restarts. Use `FAISS.save_local()` and `load_local()` if you want to persist it.
- Answers are limited to the content of your PDFs. Scanned PDFs without a text layer need OCR first.
- Embedding large document sets uses OpenAI API credits.

## Future Improvements

- Upload PDFs directly from the UI
- Persist the vector store to disk
- Add chat history and follow-up questions
- Support more file types (DOCX, TXT)

## License

MIT
