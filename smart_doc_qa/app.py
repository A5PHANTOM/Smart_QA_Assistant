import streamlit as st
import requests

# Set page configuration for wide layout resembling modern chat UI
st.set_page_config(page_title="Smart QA Assistant", page_icon="🤖", layout="wide")

# Custom CSS for a ChatGPT-like experience
st.markdown("""
<style>
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Centralize chat width, making it feel more like ChatGPT */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 800px;
    }
    
    /* Improve empty state styling */
    .empty-state {
        text-align: center;
        color: #666;
        margin-top: 4rem;
        font-family: 'Inter', sans-serif;
    }
    
    .empty-state h1 {
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .empty-state p {
        font-size: 1.1rem;
    }
    
    /* Make the chat input float better at the bottom */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Defines API address for our FastAPI backend
FASTAPI_URL = "http://127.0.0.1:5001"

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "has_document" not in st.session_state:
    st.session_state.has_document = False

# ------------- Sidebar -------------
with st.sidebar:
    st.title("🤖 Smart QA Assistant")
    st.markdown("Upload a PDF document to start asking questions about its content.")
    
    st.divider()
    
    st.subheader("📁 Document Management")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    
    if st.button("Process Document", use_container_width=True, type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing and analyzing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{FASTAPI_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"Successfully processed '{uploaded_file.name}'! You can now chat.")
                        st.session_state.has_document = True
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Backend connection failed. Is FastAPI running on port 5001? Error: {e}")
        else:
            st.warning("Please select a PDF file first.")
            
    st.divider()
    
    st.subheader("⚙️ Chat Settings")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------- Main Chat Interface -------------

# Empty state / Welcome screen when no messages exist
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class='empty-state'>
            <h1>How can I help you today?</h1>
            <p>Upload a document in the sidebar to get started.</p>
            <p>Try asking: <strong>"Summarize the document"</strong> or ask specific questions about the contents.</p>
        </div>
    """, unsafe_allow_html=True)

# Display historical chat messages
for message in st.session_state.messages:
    # Assign appropriate avatars for ChatGPT-like feel
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input at the bottom
prompt = st.chat_input("Ask a question about your document...")

if prompt:
    # Append to state and display the user's message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Process queries resolving context abstractions via cleanly routed backend mappings
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        # Visual thinking indicator
        with st.spinner("Thinking..."):
            try:
                # Issue the API request to our RAG backend
                response = requests.post(f"{FASTAPI_URL}/ask", json={"query": prompt})
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "Fallback: empty answer.")
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Backend error: {response.text}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = "Could not connect to the backend server. Please ensure the FastAPI server is running."
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
