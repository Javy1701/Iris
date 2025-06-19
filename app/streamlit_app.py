import streamlit as st
import requests
from datetime import datetime
import re

# Configure the page
st.set_page_config(
    page_title="Iris - Your Color Expert Friend",
    page_icon="🎨",
    layout="wide"
)


# --- REVISED FUNCTION ---
# This function converts only raw URLs to clickable HTML links,
# and is designed to ignore URLs already formatted in Markdown to prevent breaking them.
def convert_urls_to_links(text):
    # This regex finds raw http/https links, but uses a negative lookbehind `(?<!]\()`
    # to explicitly IGNORE URLs that are part of a markdown link like `[text](url)`.
    # This prevents the function from breaking valid markdown that st.markdown can render.
    url_pattern = re.compile(r'(?<!\]\()https?://[^\s<]+[^.,;?!<>\s]')

    # Replace only the raw URLs found. `\0` in the replacement refers to the entire matched URL.
    return url_pattern.sub(r'<a href="\0" target="_blank" rel="noopener noreferrer">\0</a>', text)


# Custom CSS for better UI
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .content {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message .avatar {
        font-size: 2rem;
        width: 40px;
        height: 40px;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .chat-message .message {
        flex: 1;
        word-wrap: break-word;
    }
    .chat-message .message a {
        color: #1E90FF; /* A nice blue color for links */
        text-decoration: underline;
    }
    .chat-message .message a:hover {
        color: #4682B4; /* Darker blue on hover */
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = 'crystal1701'  # Default user_id

# --- Sidebar for user ID input ---
with st.sidebar:
    st.title("👤 User Settings")
    st.info("Enter a User ID to start a new chat session or continue an existing one.")


    def update_user_id():
        """Callback to update user_id and clear messages."""
        if st.session_state.user_id_input != st.session_state.user_id:
            st.session_state.user_id = st.session_state.user_id_input
            st.session_state.messages = []  # Clear chat history when user changes


    st.text_input(
        "Your User ID",
        key="user_id_input",
        on_change=update_user_id,
        value=st.session_state.user_id
    )
    st.write(f"Current User: **{st.session_state.user_id}**")

# --- Main chat interface ---
st.title("🎨 Iris - Your Color Expert Friend")
st.markdown("Hi! I'm Iris, your best friend with a PhD in Color. I'm here to help you with all things color-related!")

# Display chat messages from history
for message in st.session_state.messages:
    avatar = '🎨' if message["role"] == "assistant" else '👤'
    with st.chat_message(message["role"], avatar=avatar):
        # Let st.markdown handle markdown links, and use the custom function for raw URLs.
        # The `unsafe_allow_html=True` is needed for our custom <a> tags.
        st.markdown(convert_urls_to_links(message["content"]), unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Ask me about a paint color..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar='👤'):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar='🎨'):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Iris is thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/query",
                    json={
                        "user_id": st.session_state.user_id,
                        "message": prompt
                    },
                    timeout=120
                )
                response.raise_for_status()
                full_response = response.json()["response"]
            except requests.exceptions.RequestException as e:
                full_response = f"Sorry, I could not connect to the chatbot service. Error: {e}"
            except Exception as e:
                full_response = f"An unexpected error occurred: {e}"

        # Display the final, formatted response
        message_placeholder.markdown(convert_urls_to_links(full_response), unsafe_allow_html=True)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
