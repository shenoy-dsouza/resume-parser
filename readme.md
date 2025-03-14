# Resume Parser

## Overview
This project is a **Resume Parser** that extracts key details (Name, Email, Phone Number, Experience, Skills, and Education) from a given resume file using **FAISS** for text retrieval and **Ollama** for structured text extraction.

## Features
- Loads and processes resumes in **PDF** and **text** format.
- Uses **FAISS** for fast and efficient text similarity search.
- Extracts key resume details using **Ollama AI model**.
- Implements **Sentence Transformers** for semantic search.

## Tech Stack
- **Python**
- **FAISS** (Facebook AI Similarity Search)
- **Sentence Transformers**
- **LangChain**
- **Ollama AI**
- **NumPy**

## Installation

### 1. Clone the repository:
```sh
    git clone https://github.com/shenoy-dsouza/resume-parser.git
    cd resume-parser
```

### 2. Create and activate a virtual environment:
```sh
# For macOS/Linux
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies:
```sh
    pip install -r requirements.txt
```

## Usage

### 1. Place your resume file inside the `files/` directory.

### 2. Modify the `file_path` variable in `app.py`:
```python
    file_path = "files/shenoy_dsouza.pdf"
```

### 3. Run the script:
```sh
    python app.py
```

### 4. Output Example:
```sh
Here are the extracted details in the requested format:

Name: Shenoy Dsouza  
Email: shenoy.dsouza.alt@gmail.com  
Mobile: +91 8550913482  
Experience:
- Senior Software Engineer at Myoperator (over 9 years of experience)
    • Developed backend systems for customer-centric products using PHP, Python, Django, and MySQL
    • Led a team, collaborated with stakeholders, and worked on knowledge management
- FULL STACK DEVELOPER at Genora Infotech PVT LTD
    • Involved in backend and frontend development of various websites and admin portals using server-side technologies
    • Developed and maintained REST APIs for SaaS platforms
    • Contributed to the code review process, mentored junior developers, and was responsible for Web services for mobile applications

Skills:
- Cloud platforms (AWS)
- PHP
- Python
- Django
- MySQL
- Backend systems development
- Knowledge management
- Team leadership
- Code quality assurance
- API and SQL testing methods

Education:
- Bachelor of Engineering — Computer Science, DON BOSCO COLLEGE OF ENGINEERING, Goa (2015)
```

## How It Works
1. **Load the Resume**: Reads the resume file (PDF or text).
2. **Text Splitting**: Splits text into smaller chunks using **RecursiveCharacterTextSplitter**.
3. **Embedding Generation**: Uses **Sentence Transformers** to convert text chunks into vectors.
4. **Similarity Search**: FAISS retrieves the most relevant text snippets.
5. **AI Processing**: **Ollama AI model** extracts structured information.

## Future Improvements
- Improve entity extraction using **custom-trained AI models**.
- Add support for more **file formats (DOCX, JSON, etc.)**.
- Implement a **web interface** for user-friendly interaction.

