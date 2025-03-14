import ollama
import faiss
import numpy as np
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize the embedding model for text representation
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load_and_embed_cv(file_path: str) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Loads a CV file, splits it into smaller chunks, generates embeddings, 
    and stores them in a FAISS index for efficient similarity search.
    
    Args:
        file_path (str): Path to the resume file.
    
    Returns:
        Tuple:
            - index (faiss.IndexFlatL2): FAISS index containing text embeddings.
            - texts (List[str]): List of text chunks extracted from the resume.
    """

    # Determine file type and load content
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)  # Load PDF
    else:
        loader = UnstructuredFileLoader(file_path)  # Load other text files

    documents = loader.load()  # Extract text content

    # Split text into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Convert text chunks into a list of strings
    texts = [chunk.page_content for chunk in chunks]

    # Generate embeddings for each text chunk
    embeddings = embedding_model.encode(texts).astype(np.float32)

    # Store embeddings in a FAISS index for efficient similarity search
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts


def search_text(index: faiss.IndexFlatL2, texts: List[str], query: str, top_k: int = 5) -> str:
    """
    Searches for the most relevant text chunks in the FAISS index based on a query.
    
    Args:
        index (faiss.IndexFlatL2): FAISS index containing text embeddings.
        texts (List[str]): List of text chunks.
        query (str): Search query for extracting relevant information.
        top_k (int, optional): Number of top relevant chunks to retrieve. Defaults to 5.
    
    Returns:
        str: Concatenated relevant text chunks.
    """

    # Convert query into an embedding vector
    query_embedding = embedding_model.encode([query]).astype(np.float32)

    # Perform similarity search in FAISS index
    _, indices = index.search(query_embedding, top_k)

    # Retrieve the most relevant text chunks
    retrieved_text = " ".join([texts[i] for i in indices[0]])

    return retrieved_text


def process_resume(file_path: str) -> str:
    """
    Processes a resume file to extract key details such as name, email, phone number, 
    experience, skills, and education using FAISS and an AI language model.
    
    Args:
        file_path (str): Path to the resume file.
    
    Returns:
        str: Extracted information formatted as structured text.
    """

    # Load and embed the resume text
    index, texts = load_and_embed_cv(file_path)

    # Retrieve relevant text sections based on the general query
    context = search_text(index, texts, "Extract all details from the resume.")

    # Construct a single prompt to extract all necessary details
    prompt = f"""
    Extract the following details from the provided resume text:
    
    - **Name:** Usually at the top.
    - **Email Address:** Typically follows a standard email format.
    - **Mobile Number:** Often found in the top section, starts with '+' and a country code.
    - **Work Experience:** Summarize past job roles and responsibilities.
    - **Key Skills:** Identify the candidate's main competencies.
    - **Educational Qualifications:** Mention degrees and institutions.

    Resume Text:
    {context}

    Provide the extracted details in this format:
    
    Name: [Extracted Name]  
    Email: [Extracted Email]  
    Mobile: [Extracted Phone Number]  
    Experience: [Extracted Work Experience]  
    Skills: [Extracted Key Skills]  
    Education: [Extracted Educational Qualifications]  
    """

    # Use Ollama AI model to generate a structured response
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]


# Define the path to the resume file (Replace with actual file path)
file_path = "files/shenoy_dsouza.pdf"

# Process the resume and extract relevant details
parsed_data = process_resume(file_path)

# Print the extracted details
print(parsed_data)
