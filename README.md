# Simple RAG Document Assistant

A Flask-based web application for document question-answering using Retrieval-Augmented Generation (RAG). Upload PDF documents, process them automatically, and chat with AI using either RAG mode (document context) or direct LLM mode.

## Features

- **Single Page Interface**: Clean, modern UI with all functionality on one page
- **Automatic Document Processing**: PDFs are processed immediately after upload (chunking + embeddings)
- **Real-time Processing**: Chat is disabled until document processing completes
- **Dual Mode Chat**: 
  - **RAG Mode**: Answers based on your document content
  - **Direct LLM Mode**: General AI responses without document context
- **Multiple Model Support**:
  - HuggingFace (Mistral-7B) - Free option
  - Azure OpenAI (GPT-4) - Premium option
- **Pinecone Integration**: Vector storage for efficient document retrieval
- **Real-time Status Updates**: Live progress tracking during processing

## Project Structure
chatbot/
├── app.py # Main Flask application
├── config.py # Configuration and environment variables
├── requirements.txt # Python dependencies
├── .env # Environment variables (create this file)
├── static/
│ └── style1.css # CSS styles
├── templates/
│ └── index1.html # Main HTML template
└── uploads/ # Temporary file storage

## Installation

## Create Environment and activate
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install Dependencies
pip install -r requirements.txt

## Create .env file with following keys
PINECONE_API_KEY=your_pinecone_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
AZURE_OPENAI_KEY=your_azure_openai_key_here

## Start and Run Application
python app.py


## Step-by-Step Guide
# 1. Starting the Application
bash
python app.py
Then open your browser and navigate to http://localhost:5000

# 2.Uploading a Document
Click the "Choose File" button and select a PDF document

Select your preferred Embedding Model:

HuggingFace (Free) - Uses sentence-transformers

Azure OpenAI (Premium) - More accurate but may incur costs

Click "Upload & Process Document"

# 3. Waiting for Processing
A processing overlay will appear with a spinner

The chat interface will be disabled during this time

Status indicator shows: Processing document: [filename]

Processing typically takes 1-3 minutes depending on document size

# 4. When Processing Completes
Status changes to green: Ready: [filename] (X chunks processed)

A success message appears in the chat

Chat interface becomes enabled

You're now ready to ask questions!

# 5. Chatting with Your Document
Select your preferred AI Model:

HuggingFace Mistral-7B - Free option

Azure OpenAI GPT-4 - Premium option

Choose your chat mode:

✅ RAG Enabled - Answers based on your document content

❌ RAG Disabled - General AI responses without document context

3 6. Asking Questions
Type your question in the input field

Press Enter or click "Send Message"

During AI processing:

Input field shows: "Waiting for response..."

Button shows spinner with "Processing..."

Animated typing indicator appears

Response appears with mode indicator showing: Mode: RAG | X chunks processed

# 5.1 Chat Modes Explained
🔍 RAG Mode (Recommended)
Uses your uploaded document as context

Provides specific answers based on document content

Falls back to "I don't have enough information" when answer isn't in the document

Best for: Document-specific questions, research, analysis

🤖 Direct LLM Mode
Uses AI without document context

Provides general knowledge responses

Always available, even without uploaded documents

Best for: General questions, casual conversation
