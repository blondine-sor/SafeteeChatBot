from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import *
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import openai
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
import numpy as np
from docx import Document
import PyPDF2
from dotenv import load_dotenv
import os
import json

load_dotenv()


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_word(file_path):
    """Extract text from a Word document."""
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def chunk_text(text, max_tokens=500):
    """Split text into chunks of a maximum token size."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks


def get_embeddings(text):
    """Generate embeddings for a given text using OpenAI."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response["data"][0]["embedding"], dtype="float32")



def load_documents(directory):
    """Load documents from a directory and extract text."""
    documents = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_name.endswith(".docx"):
            text = extract_text_from_word(file_path)
        else:
            continue
        documents.append({"file_name": file_name, "text": text})
    return documents


def extract_qa_pairs(text):
    """Split text into QA pairs."""
    pairs = []
    lines = text.split("\n")
    current_question = None
    current_answer = []

    for line in lines:
        if line.startswith("Q:"):
            if current_question:
                pairs.append({"question": current_question, "answer": " ".join(current_answer)})
            current_question = line[2:].strip()
            current_answer = []
        elif current_question:
            current_answer.append(line.strip())

    if current_question:  # Append the last QA pair
        pairs.append({"question": current_question, "answer": " ".join(current_answer)})

    return pairs

qa_pairs = []
for doc in documents:
    qa_pairs += extract_qa_pairs(doc["text"])


for pair in qa_pairs:
    question_embedding = get_embeddings(pair["question"])
    chunks.append({
        "question": pair["question"],
        "answer": pair["answer"],
        "embedding": question_embedding
    })

def query_qa_chatbot(user_query, top_k=3):
    """Query the chatbot using QA pairs."""
    query_embedding = get_embeddings(user_query)
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    
    # Retrieve relevant QA pairs
    relevant_qa = [chunks[i] for i in indices[0]]
    context = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in relevant_qa])
    
    # Construct prompt
    prompt = f"Here is some relevant information:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    # Get GPT response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()



def check_for_emergency(user_input):
    """Check if the user's response indicates an emergency."""
    danger_keywords = ["danger", "immediate danger", "help", "emergency", "threat", "yes","aide","urgence","unsafe","alarm","sacred"]
    user_input_lower = user_input.lower()
    for keyword in danger_keywords:
        if keyword in user_input_lower:
            return True
    return False