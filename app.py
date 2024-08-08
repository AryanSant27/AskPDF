import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import PyPDF2
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text

#Loading the Enviorment Variable
load_dotenv()

#Declare Repo_ID to select the model
repo_id="mistralai/Mistral-7B-Instruct-v0.2" 
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN') #API Key
#st.write("Hugging Face API Key:", huggingface_api_key)

# Ensure the API key is provided
if not huggingface_api_key:
    st.error("Please set the Hugging Face API token in the environment variable `HUGGINGFACEHUB_API_TOKEN`.")
else:
    embeddings = HuggingFaceEmbeddings() #used to embed words into the vector Database

    # Upload the PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf") #takes PDF Input
    if uploaded_file is not None:
        # Extract text from the uploaded PDF file
        pdf_text = PdfReader(uploaded_file)
        
        text = ''
        for page in pdf_text.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        
        chunks = text_splitter.split_text(text=text)

        # Initialize the Vector Store
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        # Use a pipeline as a high-level helper
        llm = HuggingFaceEndpoint(     #The llm object here acts as an API endpoint in which helps in accessing the LLM remotely
            repo_id=repo_id,
            max_length=128,
            temperature=0.7,
            token=huggingface_api_key
        )
        # Initialize the Conversational Retrieval Chain
        qa_chain = load_qa_chain(llm=llm, chain_type="map_reduce") #QuestionAnswer Retrieval Chain
        retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})#Here we set up the retriever Object that will help us retrive similar data

        # Create a text input box for user to ask questions
        question = st.text_input("Ask a question about the PDF")

        if question:
            # Generate answer using LangChain's Conversational Retrieval Chain
            response = qa_chain({"question": question, "input_documents": retriever.get_relevant_documents(question)})
            answer = response['output_text']
            st.write("Answer:", answer)


#Here we run the web app with help of API Endpoint but we can also run the Model Locally by downloading. That way we wont have to connect with internet and
#it will run a lot faster locally.

#To use that model locally we must use Model Name and Tokeninzer for EG:

#LOAD MODEL AND TOKENINZER

#"model=AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")"
#"tokenizer=AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")"

#Also dont forget to import important libraries like:
#"from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline"

#while tring to run the LLM locally we need to import HuggingFacePipeline and declare an hf object which tells the type of work for our Model
#CREATE A TEXT GEN PIPELINE:
#pipe = pipeline(
#  "text-generation",
#   model=model,
#   tokenizer=tokenizer,
#  max_new_tokens=512,
#   temperature=0.7,
#   top_p=0.95,
#   repetition_penalty=1.15
#)

#CREATE A LANGCHAIN OBJECT:
#llm = HuggingFacePipeline(pipeline=pipe)

#After this the steps are same. Use this llm object in qa_chain and then pass that into the response method.
