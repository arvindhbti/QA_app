**QA app using Streamlit**

**RAG Pipeline** -

step1 - Extract data from source
step2 - Chunks the documents
step3 - Embeddings of chunks
step4 - Indexing and storing in vector database

**Retriever** -

step1 - Extract relevant data from vector database

**Generation** -

step1 - Generate response using relevant text and user query



**STEPS TO DIPLOY THE APPLICATION IN CLOUD**

1)  Set Project ID Environment Variable

2) Build the docker image

3) Upload the container image

4) Create Cluster

5) Deploy Application

6) Expose your application to the internet

7) Check Service

8) See the app in action on web address

**STEPS TO RUN THE APP in local** - 
1) Open command prompt
2) Run as "streamlit run summ_app.py"
3) It will be hosted in your local browser
