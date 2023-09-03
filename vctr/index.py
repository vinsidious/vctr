import streamlit as st
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, LLMPredictor, ServiceContext, SimpleDirectoryReader

LLM_INDEX_PATH = '/Users/vince/vctr/llm_index.json'
VBT_INDEX_PATH = '/Users/vince/vctr/vbt_index.json'


def load_index():
    return GPTSimpleVectorIndex.load_from_disk(LLM_INDEX_PATH)


index = load_index()

st.title('VBT Documentation')
query = st.text_input('What would you like to ask?', '')

if st.button('Submit'):
    response = index.query(query)
    st.markdown(response)
