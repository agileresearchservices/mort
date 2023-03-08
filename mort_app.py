import streamlit as st
from uuid import uuid4
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

OPENAI_API_KEY =  st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY =  st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV =  st.secrets["PINECONE_API_ENV"]

def get_answer(query):
    query = "Answer this question from the perspective of a mortgage broker in a training session. Be consice. Answer the question in markdown format and use bullet points and paragraphs whenever needed. " + query
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index_name = "aimortgageapp"
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    llm = OpenAI(temperature=0.7, max_tokens=512 ,openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(query, include_metadata=True)
    answer = chain.run(input_documents=docs, question=query)
    return answer

st.title("Mort Q&A Application")

query = st.text_input("Enter your question here")

if query:
    answer = get_answer(query)
    st.write("**Question**: ", query)
    st.write("**Answer**: ", answer)