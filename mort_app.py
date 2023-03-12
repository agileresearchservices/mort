import streamlit as st
from uuid import uuid4
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime

OPENAI_API_KEY =  st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY =  st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV =  st.secrets["PINECONE_API_ENV"]

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "aimortgageapp"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
llm = OpenAI(temperature=0.3, max_tokens=512 ,openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
resources = []

def get_answer(query):
    query = "Answer this question from the perspective of a mortgage broker in a training session. Be consice. If you don't know the answer from the context, refer to general Mortgage, Banking, and Real Estate industry information. If you find foul language, please rewrite it in a professional way." + query
    docs = docsearch.similarity_search(query, include_metadata=True, k=20)
    answer = chain.run(input_documents=docs, question=query)
    return answer, docs


st.title("Mort, Your Brokerage AI Assistant")
query = st.text_input("Enter your question here")


if query:
    with st.spinner(f"Thinking..."):
        answer, docs = get_answer(query)
    st.write("## Mort Says: ")
    st.write(answer)
    st.markdown('---')
    st.write("## Additional Resources:")

    # unique_docs = {}
    # for doc in docs:
    #     title = str(doc.metadata.get('title'))
    #     short_description = str(doc.metadata.get('short_description'))
    #     child_url = str(doc.metadata.get('url_child'))
    #     parent_url = str(doc.metadata.get('url_parent'))
    #     timestamp = datetime.strptime(str(doc.metadata.get('timestamp')), "%Y-%m-%d %H:%M:%S").strftime("%H:%M:%S")
    #     key = title + '||' + short_description
    #     if key not in unique_docs:
    #         unique_docs[key] = {'child_urls': [(child_url, timestamp)],
    #                             'parent_url': parent_url}
    #     else:
    #         unique_docs[key]['child_urls'].append((child_url, timestamp))

    # for title_desc, doc_info in unique_docs.items():
    #     st.write('## ' + title_desc.split('||')[0])
    #     st.write('**Short Description:** ' + title_desc.split('||')[1])
    #     st.markdown(f"[Start from beginning]({doc_info['parent_url']})")
    #     st.write('**Jump to Moments:** ')
    #     child_urls = []
    #     for child_url, timestamp in doc_info['child_urls']:
    #         child_urls.append(f"[{timestamp}]({child_url})")
    #     st.write(', '.join(child_urls))
    # unique_docs = {}
    # for doc in docs:
    #     title = str(doc.metadata.get('title'))
    #     short_description = str(doc.metadata.get('short_description'))
    #     child_url = str(doc.metadata.get('url_child'))
    #     parent_url = str(doc.metadata.get('url_parent'))
    #     timestamp = datetime.strptime(str(doc.metadata.get('timestamp')), "%Y-%m-%d %H:%M:%S").strftime("%H:%M:%S")
    #     key = title + '||' + short_description
    #     if key not in unique_docs:
    #         unique_docs[key] = {'child_urls': {child_url: {timestamp}},
    #                             'parent_url': parent_url}
    #     else:
    #         if child_url not in unique_docs[key]['child_urls']:
    #             unique_docs[key]['child_urls'][child_url] = {timestamp}
    #         else:
    #             unique_docs[key]['child_urls'][child_url].add(timestamp)

    # for title_desc, doc_info in unique_docs.items():
    #     st.write('## ' + title_desc.split('||')[0])
    #     st.write('**Short Description:** ' + title_desc.split('||')[1])
    #     st.markdown(f"[Start from beginning]({doc_info['parent_url']})")
    #     st.write('**Jump to Moments:** ')
    #     child_urls = []
    #     for child_url, timestamps in doc_info['child_urls'].items():
    #         sorted_timestamps = sorted(timestamps, key=lambda x: datetime.strptime(x, "%H:%M:%S"))
    #         child_urls.append(f"[{', '.join(sorted_timestamps)}]({child_url})")
    #     st.write(', '.join(child_urls))

    unique_docs = {}
    for doc in docs:
        title = str(doc.metadata.get('title'))
        short_description = str(doc.metadata.get('short_description'))
        child_url = str(doc.metadata.get('url_child'))
        parent_url = str(doc.metadata.get('url_parent'))
        timestamp = datetime.strptime(str(doc.metadata.get('timestamp')), "%Y-%m-%d %H:%M:%S").time()
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
        st.write('## ' + title_desc.split('||')[0])
        st.write('**Short Description:** ' + title_desc.split('||')[1])
        st.markdown(f"[Start from beginning]({doc_info['parent_url']})")
        st.write('**Jump to Moments:** ')
        child_urls = []
        for child_url, timestamps in doc_info['child_urls'].items():
            sorted_timestamps = sorted(timestamps)
            sorted_timestamp_strings = [ts.strftime("%H:%M:%S") for ts in sorted_timestamps]
            child_urls.append(f"[{', '.join(sorted_timestamp_strings)}]({child_url})")
        st.write(', '.join(child_urls))