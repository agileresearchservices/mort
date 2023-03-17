import streamlit as st
from uuid import uuid4
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime
import re

OPENAI_API_KEY =  st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY =  st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV =  st.secrets["PINECONE_API_ENV"]

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "aimortgageapp"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
llm = OpenAI(temperature=0.3, max_tokens=1024 ,openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
resources = []

def clean_output(output: str) -> str:
    # Remove special characters
    cleaned_output = re.sub(r'[^a-zA-Z0-9,.:;?!()%\- \n]', '', output)

    # Add spaces after punctuation marks
    cleaned_output = re.sub(r'([.,:;?!()\-])(\w)', r'\1 \2', cleaned_output)

    return cleaned_output

def get_answer(query):
    query = "Answer this question from the perspective of a mortgage broker in a training session. Be consice. If you don't know the answer from the context, refer to general Mortgage, Banking, and Real Estate industry information. If you find foul language, please rewrite it in a professional way." + query
    docs = docsearch.similarity_search(query, include_metadata=True, k=20)
    answer = chain.run(input_documents=docs, question=query)
    return answer, docs


st.title("Mort, Your Brokerage AI Assistant")
st.write("I am your virtual assistant. You can ask me questions or, I can perform tasks such as writing an email and other administrative tasks.")
st.write("Here a few things to try:")
st.write(" - Write an email following up with a customer to set up a meeting about their upcoming mortgage closing")
st.write(" - Have a specific client scenario you need help with? Ask me for advice on how to structure the loan and find the best lender for their needs.")
st.write(" - Need to explain a complex mortgage concept to a client? Ask me for tips on how to simplify it and make it more understandable.")


query = st.text_input("Enter your question here")


if query:
    with st.spinner(f"Thinking..."):
        answer, docs = get_answer(query)
    st.write("## Mort Says: ")
    st.markdown(clean_output(answer))
    st.markdown('---')
    st.write("## Additional Resources:")

    unique_docs = {}
    for doc in docs:
        title = str(doc.metadata.get('title'))
        short_description = str(doc.metadata.get('short_description'))
        child_url = str(doc.metadata.get('url_child'))
        parent_url = str(doc.metadata.get('url_parent'))
        try:
            timestamp = datetime.strptime(str(doc.metadata.get('timestamp')), "%Y-%m-%d %H:%M:%S").strftime("%H:%M:%S")
        except ValueError:
            timestamp = "00:00:00"
        key = title + '||' + short_description
        if key not in unique_docs:
            unique_docs[key] = {'child_urls': {child_url: {timestamp}},
                                'parent_url': parent_url}
        else:
            if child_url not in unique_docs[key]['child_urls']:
                unique_docs[key]['child_urls'][child_url] = {timestamp}
            else:
                unique_docs[key]['child_urls'][child_url].add(timestamp)

    for title_desc, doc_info in unique_docs.items():
        title, short_desc = title_desc.split('||')

        if not title or not short_desc:
            continue

        cleaned_short_desc = clean_output(short_desc)

        st.markdown(f"### [{title}]({doc_info['parent_url']})")
        st.markdown(cleaned_short_desc)
        st.markdown(f"[Start from beginning]({doc_info['parent_url']})")
        st.markdown('**Jump to Moments:** ')
        child_urls = []
        for child_url, timestamps in doc_info['child_urls'].items():
            child_urls.append(f"[{', '.join(timestamps)}]({child_url})")
        st.write(', '.join(child_urls))

