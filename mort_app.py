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
MAX_RETRIES = 3

def clean_output(output: str) -> str:
    # Remove special characters
    cleaned_output = re.sub(r'[^a-zA-Z0-9,.:;?!()%/\- \n]', '', output)

    # Add spaces after punctuation marks
    cleaned_output = re.sub(r'([.,:;?!()\-])(\w)', r'\1 \2', cleaned_output)

    return cleaned_output

def get_answer(query):
    query = "Answer the question based on the context below. You should follow ALL the following rules when generating and answer: " \
        "- Be concise in your response " \
        "- If the context doesn't provide the necessary information, use general Mortgage, Banking, and Real Estate industry knowledge. " \
        "- Rewrite any inappropriate language professionally. " \
        "- If the question asks specifically about current rates, lender rates, or interest rates, only refer them to https://www.ratehub.ca. NEVER append anything to the hyperlink. NEVER display any current rates, lender rates, or interest rates." \
        "- Use bullet points, lists, paragraphs and text styling to present the answer in markdown format." \
        + query
    docs = docsearch.similarity_search(query, include_metadata=True, k=20)
    answer = chain.run(input_documents=docs, question=query)
    return answer, docs

st.title("Mort, Your Brokerage AI Assistant")
st.write("I am your virtual assistant. You can ask me questions or, I can perform tasks such as writing an email and other administrative tasks.")
st.write("Here a few things to try:")
st.write(" - Ask me to write an email following up with a customer to set up a meeting about their upcoming mortgage closing")
st.write(" - Suggest closing strategies for a bullish customer")
st.write(" - Need to explain a complex mortgage concept to a client? Ask me for tips on how to simplify it and make it more understandable.")


query = st.text_input("Enter your question here")

def process_query(query):
    for retry_count in range(MAX_RETRIES):
        try:
            with st.spinner(f"Thinking..."):
                answer, docs = get_answer(query)
            return answer, docs
        except Exception as e:
            st.warning(f"An error occurred while processing your query: {e}")
            if retry_count < (MAX_RETRIES - 1):
                st.warning(f"Retrying... (Attempt {retry_count + 1} of {MAX_RETRIES})")
                time.sleep(2)  # Adding a small delay before retrying
            else:
                st.error(f"Failed to process your query after {MAX_RETRIES} attempts. Please try again later.")
                break
    return None, None

if query:
    answer, docs = process_query(query)
    if answer and docs:
        st.write("## Mort Says: ")
        st.markdown(answer)
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


html_code = """
<a href="https://mort-ai.streamlit.app" border="0" style="cursor:default" rel="nofollow">
    <img src="https://chart.googleapis.com/chart?cht=qr&chl=https%3A%2F%2Fmort-ai.streamlit.app&chs=180x180&choe=UTF-8&chld=L|2">
</a>
"""

css_code = """
<style>
    #my-container {
        position: fixed;
        top: 0;
        right: 0;
        padding: 10px;
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 5px;
        display: none;
    }
    @media screen and (min-width: 768px) {
        #my-container {
            display: block;
        }
    }
</style>
"""

with st.container():
    st.markdown("### Take me with you!")
    st.markdown(html_code + css_code, unsafe_allow_html=True)