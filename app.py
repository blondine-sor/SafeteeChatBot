from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_openai.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import uvicorn
import jq
import json
from dotenv import load_dotenv
import os

load_dotenv()


# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API Key
API_KEY = os.getenv("OPENAI_APIKEY") 

#Chatbot name
CHATBOT_NAME = "Safetee"

# Emergency contact function
def make_emergency_call(user_id, username):
    
    print(f"Emergency call initiated for user: {username} (ID: {user_id})")
    return {"response": {"answer":f"{username}, You can make an emergency call if you logged in with the emergency button on the upper right side of your screen."}}


# Load QA document and create vector store
def load_qa_documents(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        qa_data = json.load(file)
    documents = []
    for entry in qa_data:
        text = f"Question: {entry['question']}\nAnswer: {entry['answer']}"
        documents.append(Document(page_content=text))
    return documents


loader = JSONLoader(
    "qa_doc.json",
    jq_schema=".[] | {page_content: (.question + \"\\nAnswer: \" + .answer), metadata: {}}"
)
documents = load_qa_documents("qa_doc.json")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
vector_store = FAISS.from_documents(documents, embeddings)
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0,openai_api_key=API_KEY),
    retriever=vector_store.as_retriever()
)

# Input schema
class UserQuery(BaseModel):
    username: str
    user_id: int 
    query: str

# List of words indicating harm
HARM_INDICATORS = [
    "unsafe", "threat", "danger", "harm", "violence", "attack", "hurt", "kill", "injure", "abuse", "assault",
    "bullying", "exploit", "crisis", "panic", "emergency", "trauma", "fear", "hostile", "alarm",
    "I feel threatened", "I am in danger", "Someone is following me", "I’m not safe", "Please help me",
    "They’re going to hurt me", "I’m being attacked", "I’m scared", "This is an emergency", "I need help now"
]

# Check if query contains harm indicators
def check_for_harm_indicators(query):
    for indicator in HARM_INDICATORS:
        if indicator.lower() in query.lower():
            return True
    return False

# API route for chatbot
@app.post("/chat")
async def chat_with_bot(user_query: UserQuery):
    username = user_query.username
    user_id = user_query.user_id
    query = user_query.query

    # Check for emergency situations
    if check_for_harm_indicators(query):
        return make_emergency_call(user_id,username)

    # Use LangChain retrieval chain to find a response
    try:
        response = retrieval_chain.invoke({"question": query, "chat_history": []})
        return {"response":response}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error generating response.")








if __name__=="__name__":
    uvicorn.run(app,host="0.0.0.0", port=8060,workers=1)