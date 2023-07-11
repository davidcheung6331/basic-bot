# Imports
import os 
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message



st.set_page_config(
    page_title="Demo Page",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
# image = Image.open("stock.png")
# st.image(image, caption='created by MJ')





system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key

if 'question' not in st.session_state:
    st.session_state['question'] = []
if 'answer' not in st.session_state:
    st.session_state['answer'] = []




st.title('🦜 Query your PDF document ')
prompt = st.text_input("Enter your question to query your PDF documents","")
if prompt:
    model_id = "gpt-3.5-turbo"
    loaders = PyPDFLoader('case.pdf')
    index = VectorstoreIndexCreator().from_loaders([loaders])


    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM
    llm = ChatOpenAI(model_name=model_id, temperature=0.2)
    response = index.query(llm=llm, question=prompt, chain_type='stuff')

    
    # Add the question and the answer to display chat history in a list
    # Latest answer appears at the top
    st.session_state.question.insert(0,prompt  )
    st.session_state.answer.insert(0,response  )
    
    # Display the chat history
    for i in range(len( st.session_state.question)) :
        message(st.session_state['question'][i], is_user=True)
        message(st.session_state['answer'][i], is_user=False)
