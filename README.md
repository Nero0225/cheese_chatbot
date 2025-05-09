# Cheese Data Chatbot

This project implements a RAG (Retrieval Augmented Generation) chatbot to answer questions about cheese products based on scraped data.

## Features

- Scrapes cheese data from a specified website (data provided in `scraped_cheese_data.json`).
- Uses Pinecone as a vector database to store and query cheese data embeddings.
- Leverages OpenAI's GPT-4o for language understanding and response generation.
- Provides a user-friendly chat interface built with Streamlit.
- Displays context data (retrieved cheese information) used to answer questions.

## Setup

1.  **Clone the repository (if applicable).**
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    Create a `.env` file in the project root by copying `.env.example`:
    ```bash
    cp .env.example .env
    ```
    Then, fill in your actual API keys and Pinecone details in the `.env` file:
    - `OPENAI_API_KEY`: Your OpenAI API key.
    - `PINECONE_API_KEY`: Your Pinecone API key.
    - `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., "gcp-starter", "us-west1-gcp").
    - `PINECONE_INDEX_NAME`: (Optional) Desired name for your Pinecone index (defaults to "cheese-chatbot-index").
    - `EMBEDDING_MODEL`: (Optional) OpenAI embedding model to use (defaults to "text-embedding-3-small").
    - `LLM_MODEL`: (Optional) OpenAI LLM model to use (defaults to "gpt-4o").

## Usage

1.  **Ingest data into Pinecone:**
    Run the data ingestion script. This only needs to be done once, or when the source data changes.
    ```bash
    python ingest_data.py
    ```
2.  **Run the Streamlit chatbot application:**
    ```bash
    streamlit run chatbot/app.py
    ```
    Open your browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Project Structure

```
/your_project_root
|-- .env                 # Local environment variables (API keys, etc.) - DO NOT COMMIT
|-- .env.example         # Example environment file
|-- requirements.txt     # Python dependencies
|-- README.md            # This file
|-- ingest_data.py       # Script to process data and load into Pinecone
|-- scraped_cheese_data.json # Provided scraped cheese data
|-- chatbot/
|   |-- app.py           # Main Streamlit application file
|   |-- utils.py         # Helper functions for OpenAI, Pinecone, etc.
``` 