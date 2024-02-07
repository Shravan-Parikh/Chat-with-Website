import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader

# response from the model
def get_response(user_input):
    return "hey, shravan I don't know"

# App - Cnnfig
st.set_page_config(page_title="Chat with Website", page_icon="robot")
st.title("Chat with Websites")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am possible bot. How can I help You"),
    ]

#sidebar
with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Website URL")

if website_url is None or website_url=="":
    st.info(f"Please Enter a valid URL to Enable and Use the Bot")
   
else :
    #user input logic
    user_query=st.chat_input("Type your question here...")

    if user_query is not None and user_query != "":
        response=get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)