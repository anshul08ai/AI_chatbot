from flask import Flask, render_template, request, jsonify
import os
import uuid
import threading
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Global variables to track processing state
processing_in_progress = False
vector_store_ready = False
current_filename = None
chunks_processed = 0

class DocumentProcessor:
    def __init__(self):
        self.upload_folder = Config.UPLOAD_FOLDER
        self.allowed_extensions = Config.ALLOWED_EXTENSIONS
        
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def save_uploaded_file(self, file):
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
        filename = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join(self.upload_folder, filename)
        file.save(file_path)
        return file_path, file.filename
    
    def process_pdf(self, file_path):
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = text_splitter.split_documents(documents)
            
            # Clean up the file after processing
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return chunks
        except Exception as e:
            # Clean up file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

class VectorStoreManager:
    def __init__(self):
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        self.vector_store = None
        
    def initialize_index(self, dimension=768):
        # Delete existing index if it exists
        if self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)
            
        # Create new index
        self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
        )
        return self.pc.Index(self.index_name)
    
    def get_embeddings(self, model_type="huggingface"):
        if model_type == "huggingface":
            return HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-mpnet-base-v2",
                huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY
            )
        elif model_type == "azure":
            return AzureOpenAIEmbeddings(
                azure_deployment='text-embedding-3-small',
                api_key=Config.AZURE_OPENAI_KEY,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
            )
    
    def get_vector_store(self, embeddings):
        index = self.initialize_index()
        self.vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        return self.vector_store
    
    def add_documents(self, documents, embeddings_model="huggingface"):
        embeddings = self.get_embeddings(embeddings_model)
        vector_store = self.get_vector_store(embeddings)
        uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
        return len(documents), vector_store

def get_llm(model_type):
    if model_type == "huggingface":
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="conversational",
            temperature=0.5,
            huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY
        )
        return ChatHuggingFace(llm=llm)
    elif model_type == "azure":
        return AzureChatOpenAI(
            azure_deployment='gpt-4o',
            temperature=0.5,
            api_key=Config.AZURE_OPENAI_KEY,
            api_version='2023-03-15-preview',
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
        )

def process_document_async(file_path, model_type, filename):
    """Process document in background thread"""
    global processing_in_progress, vector_store_ready, current_filename, chunks_processed
    
    try:
        processing_in_progress = True
        current_filename = filename
        
        doc_processor = DocumentProcessor()
        vector_manager = VectorStoreManager()
        
        # Process the document
        chunks = doc_processor.process_pdf(file_path)
        
        # Add to vector store
        chunks_count, vector_store = vector_manager.add_documents(chunks, model_type)
        chunks_processed = chunks_count
        
        # Update global state
        vector_store_ready = True
        processing_in_progress = False
        
        print(f"Successfully processed {chunks_count} chunks from {filename}")
        
    except Exception as e:
        processing_in_progress = False
        vector_store_ready = False
        current_filename = None
        print(f"Error processing document: {str(e)}")

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    global processing_in_progress
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        model_type = request.form.get('model_type', 'huggingface')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        doc_processor = DocumentProcessor()
        if not doc_processor.allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        if processing_in_progress:
            return jsonify({'error': 'Another document is currently being processed'}), 400
        
        # Save the file
        file_path, original_filename = doc_processor.save_uploaded_file(file)
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_document_async,
            args=(file_path, model_type, original_filename)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Document upload and processing started',
            'filename': original_filename,
            'status': 'processing'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global processing_in_progress, vector_store_ready
    
    try:
        data = request.get_json()
        question = data.get('question')
        model_type = data.get('model_type', 'huggingface')
        use_rag = data.get('use_rag', False)
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Check if processing is still in progress
        if processing_in_progress:
            return jsonify({'error': 'Document is still being processed. Please wait...'}), 400
        
        # Check if RAG is requested but no document is ready
        if use_rag and not vector_store_ready:
            return jsonify({'error': 'No document has been processed yet. Please upload and process a document first.'}), 400
        
        llm = get_llm(model_type)
        
        if use_rag and vector_store_ready:
            # RAG mode - get context from documents
            vector_manager = VectorStoreManager()
            embeddings = vector_manager.get_embeddings(model_type)
            vector_store = vector_manager.get_vector_store(embeddings)
            
            retriever = vector_store.as_retriever()
            retrieved_docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            system_prompt = f"""You are a helpful assistant. Answer the question based only on the following context.
If the answer is not in the context, say "I don't have enough information in the provided document."

Context:
{context}"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]
            
            result = llm.invoke(messages)
            
            return jsonify({
                'answer': result.content,
                'mode': 'RAG',
                'chunks_processed': chunks_processed,
                'document_name': current_filename
            })
        else:
            # Direct LLM mode
            messages = [
                SystemMessage(content="You are a helpful assistant. Answer the user's question directly."),
                HumanMessage(content=question),
            ]
            
            result = llm.invoke(messages)
            
            return jsonify({
                'answer': result.content,
                'mode': 'Direct LLM'
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    global processing_in_progress, vector_store_ready, current_filename, chunks_processed
    
    status_info = {
        'processing_in_progress': processing_in_progress,
        'vector_store_ready': vector_store_ready,
        'filename': current_filename,
        'chunks_processed': chunks_processed
    }
    
    return jsonify(status_info)

@app.route('/clear', methods=['POST'])
def clear_documents():
    """Clear the current document and reset state"""
    global processing_in_progress, vector_store_ready, current_filename, chunks_processed
    
    processing_in_progress = False
    vector_store_ready = False
    current_filename = None
    chunks_processed = 0
    
    return jsonify({
        'message': 'Document state cleared',
        'status': 'ready_for_new_document'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)