from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from Bio import Entrez
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Set up PubMed API
Entrez.email = "lpinto1204@gmail.com"

# Load OpenAI Models
embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5, max_tokens=200)

# Initialize Memory
memory = ConversationBufferMemory(memory_key="pubmed_memory", return_messages=True)

def search_pubmed(query, max_results=3):
    """Search PubMed for articles related to a query and finding mode details."""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        pmid_list = record["IdList"]
        
        articles = []
        for pmid in pmid_list:
            fetch_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
            abstract = fetch_handle.read()
            articles.append({"pmid": pmid, "abstract": abstract})

        return articles

    except Exception as e:
        return [{"pmid": "N/A", "abstract": f"Error fetching PubMed articles: {str(e)}"}]

# Streamlit UI
st.title("🩺 Medical Chatbot with PubMed Integration")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
# User Input
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Store user query in memory
    st.chat_message("user").write(user_input)

    # Step 1: Generate Hypothetical Answer
    with st.spinner("🤖 Generating medical response..."):
        hypothetical_doc = llm.predict(f"Generate a detailed medical explanation for: {user_input}")

    # Step 2: Search PubMed
    st.chat_message("assistant").write("🔍 Searching PubMed for relevant articles...")
    pubmed_results = search_pubmed(user_input)

    # Step 3: Summarize and Display Results
    response_text = f"🤖 **Medical Answer:**\n{hypothetical_doc}\n\n"
    response_text += "📚 **Relevant Research Papers:**\n"
    
    for idx, article in enumerate(pubmed_results):
        response_text += f"🔹 **Article {idx+1}**\n"
        response_text += f"🆔 PMID: {article['pmid']}\n"
        response_text += f"📜 Abstract: {article['abstract'][:300]}...\n"
        response_text += f"🔗 [Read more](https://pubmed.ncbi.nlm.nih.gov/{article['pmid']})\n\n"

    # Display Response
    st.chat_message("assistant").write(response_text)

    # Store in conversation memory
    memory.save_context({"user": user_input}, {"output": response_text})

    # Store messages in session
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response_text})
