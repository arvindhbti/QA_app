# Import Packages
import streamlit as st
import os
import logging
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from io import StringIO
import nltk

## Load .md file using Langchain markdown Loader
def ingest_pdf(markdown_path):
    """Load the ingested file"""
    
    loader = UnstructuredMarkdownLoader(markdown_path)
    data = loader.load()

    logging.info("md file loaded successfully.")
    return data 
    

## Split the PDF file using Langchain RecursiveCharacterTextSplitter
def split_documents(data):
    """Split documents into smaller chunks."""
    
    document = data[0].page_content
    markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=50)
    docs = markdown_splitter.create_documents([document])
    
    logging.info("Documents split into chunks.")
    return docs 


def load_vector_db(docs):
    """Load or create the vector database."""
    
    # Pull the embedding model if not already available
    # loading the embedding model from huggingface
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(
      model_name=embedding_model_name,
      model_kwargs=model_kwargs
    )
       
    #loading the data and correspond embedding into the FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    logging.info("Vector database created.")
    return vectorstore

def create_retriever(vector_db):
    """Create a retriever."""
    
    #creating a retriever on top of database
    retriever = vector_db.as_retriever()

    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    """Create the chain"""    
    
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
  
    prompt = ChatPromptTemplate.from_template(template)

    # create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created")
    return chain


def main():
    
    # title of application
    st.title("RAG QA Application")
    
    # User question
    question = st.text_input("Enter your question:", "")
    # Upload .md file
    uploaded_file = st.file_uploader("Choose a file")
    # choose path to store file
    markdown_path = "./uploaded_file.md"
    # write data in md file
    if uploaded_file is not None:
        with open(markdown_path, "wb") as f:
            f.write(uploaded_file.getbuffer())


    if question:
        if st.button("Generate Output"):
            with st.spinner("Generating response..."):
                try:                
                    #question = "what is the effective date in the given text"

                    # upload .md files  
                    data = ingest_pdf(markdown_path)

                    # chunk the documents
                    docs = split_documents(data)

                    # create embeddings of the chunks and store in the vector database
                    vectorstore = load_vector_db(docs)

                    # create retriever
                    retriever = create_retriever(vectorstore)

                    # create chain
                    # Initialize an instance of the Ollama model
                    llm = OllamaLLM(model="phi")
                    chain = create_chain(retriever, llm)

                    # generate response
                    response = chain.invoke(input=question)

                    # Set header
                    st.subheader("Output:")
                    # write response
                    st.write(response)
                    os.remove(markdown_path)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    else:
        st.info("Please enter a question to get started.")
    
    
if __name__ == "__main__":
    main()