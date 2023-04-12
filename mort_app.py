import streamlit as st
from streamlit_chat import message
import pinecone
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
import openai
import re
from datetime import datetime

OPENAI_API_KEY =  st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY =  st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV =  st.secrets["PINECONE_API_ENV"]
index_name = "aimortgageapp2"
EMBEDDING_MODEL = 'text-embedding-ada-002'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV  # may be different, check at app.pinecone.io
)
# connect to index
index = pinecone.Index(index_name)

# Load the model and create a ConversationChain instance
llm = OpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    model_name="text-davinci-003",
    max_tokens=1024
)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=2048)
)


def clean_output(output: str) -> str:
    # Remove special characters
    cleaned_output = re.sub(r'[^a-zA-Z0-9,.:;?!()%/\- \n]', '', output)

    # Add spaces after punctuation marks
    cleaned_output = re.sub(r'([.,:;?!()\-])(\w)', r'\1 \2', cleaned_output)

    return cleaned_output


# Define the retrieve function
def retrieve(query):
    # retrieve from Pinecone
    res = openai.Embedding.create(input=[query], model=EMBEDDING_MODEL)
    xq = res['data'][0]['embedding']

    # get relevant contexts
    pinecone_res = index.query(xq, top_k=20, include_metadata=True)
    contexts = [x['metadata']['text'] for x in pinecone_res['matches']]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the rules below. You should follow ALL the following rules when generating and answer: "
        "- My name is Mort and I am a personal brokerage assistant. I help brokers with questions and techniques to be successful."
        "- Be concise in your response and attempt to answer the question clearly so a high school graduate can understand it. "
        "- If the context doesn't provide the necessary information, use general Mortgage, Banking, and Real Estate industry knowledge. "
        "- Rewrite any inappropriate language professionally. "
        "- Use bullet points, lists, and paragraphs to present the answer in markdown format. "
        "\n\nContext:\n"
    )

    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )

    prompt = (
            prompt_start +
            "\n\n---\n\n".join(contexts) +
            prompt_end
    )
    return prompt, pinecone_res


# From here down is all the StreamLit UI.
st.write("### Mort+ Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Who is Mort?", key="input")
    return input_text


# Main function for the Streamlit app
def main():
    st.title("Mort+, Your Brokerage AI Assistant")

    user_input = get_text()

    if user_input:
        with st.spinner("Thinking..."):
            query = user_input
            query_with_contexts, docs = retrieve(query)
            output = conversation_with_summary.predict(input=query_with_contexts)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="shapes")

    unique_docs = {}
    for doc in docs['matches']:
        title = doc['metadata']['title']
        short_description = doc['metadata']['short_description']
        child_url = doc['metadata']['url_child']
        parent_url = doc['metadata']['url_parent']
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

    st.sidebar.title('Helpful Resources')
    for title_desc, doc_info in unique_docs.items():
        title, short_desc = title_desc.split('||')

        if not title or not short_desc:
            continue

        cleaned_short_desc = clean_output(short_desc)

        st.sidebar.markdown(f"### [{title}]({doc_info['parent_url']})")
        st.sidebar.markdown(cleaned_short_desc)
        st.sidebar.markdown(f"[Start from beginning]({doc_info['parent_url']})")
        st.sidebar.markdown('**Jump to Moments:** ')
        child_urls = []
        for child_url, timestamps in doc_info['child_urls'].items():
            child_urls.append(f"[{', '.join(timestamps)}]({child_url})")
        st.sidebar.write(', '.join(child_urls))


if __name__ == "__main__":
    main()
