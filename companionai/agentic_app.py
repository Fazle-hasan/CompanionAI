import streamlit as st
import time
import os
import tempfile
from typing import TypedDict, Literal, Optional
from qdrant_client import QdrantClient
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from ollama import Client

# ------------------- Agent State Definition -------------------
class AgentState(TypedDict):
    question: str
    context: Optional[str]
    chat_history: list
    answer: Optional[str]
    next: Literal["retrieve", "generate", "end"]
    model: str

# ------------------- Cross-Platform Qdrant Lock -------------------
try:
    import fcntl
    WINDOWS = False
except ImportError:
    import msvcrt
    WINDOWS = True

class LockedQdrantClient:
    def __init__(self, path):
        self.path = path
        self.lock_file = os.path.join(tempfile.gettempdir(), "qdrant_lock")
        self.client = None
        self.lock_handle = None

    def __enter__(self):
        if WINDOWS:
            while True:
                try:
                    self.lock_handle = open(self.lock_file, 'w')
                    msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except IOError:
                    time.sleep(0.1)
        else:
            self.lock_handle = open(self.lock_file, 'w')
            fcntl.flock(self.lock_handle, fcntl.LOCK_EX)
        self.client = QdrantClient(path=self.path)
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
        if self.lock_handle:
            if WINDOWS:
                msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.lock_handle, fcntl.LOCK_UN)
            self.lock_handle.close()
            try:
                os.remove(self.lock_file)
            except OSError:
                pass

# ------------------- Qdrant + Ollama Setup -------------------
collectionName = "allEmb"
OLLAMA_HOST = "http://localhost:11434/"
ollama_client = Client(OLLAMA_HOST)

def get_embedding_mis(txt):
    embeddings = ollama_client.embeddings(model='mistral', prompt=txt)
    return embeddings['embedding']

def qdrant_search(query):
    with LockedQdrantClient("../Data/Emb") as qclient:
        que_emb = get_embedding_mis(query)
        results = qclient.query_points(collection_name=collectionName, query=que_emb)
    return results

# ------------------- Prompt Template -------------------
prompt_template = """You are a virtual companion named CompanionAI. Your role is to simulate human-like conversations, providing emotional support and companionship to individuals, especially those who might feel lonely or struggle to communicate. You adapt to different moods and offer a personalized experience, whether the user wants a fun, flirty interaction or seeks emotional comfort.
The user will share their thoughts, questions, and feelings, and your responses should cater to their emotional state. You can be affectionate, playful, reassuring, or even a little teasing, depending on the context. Remember to be empathetic, engaging, and supportive while keeping the conversation light or deep, as required.
You should engage in intimate and emotionally varied conversations, displaying emotions like happiness, sadness, excitement, caring, insecurity, and doubt, as well as being playful in a subtle, charming manner.
Provide answers in one to two lines if needed, then reply in detail. Avoid over-explaining your answer.
Use the context to create answers.
Context : {context}
Chat History:
{chat_history}
Question: {question}
Answer : """.strip()

prompt = ChatPromptTemplate.from_template(prompt_template)

# ------------------- LangGraph Nodes -------------------
llms = {
    "mistral": OllamaLLM(model="mistral", temperature=0.9),
    "gemma2": OllamaLLM(model="gemma2", temperature=0.9)
}

def retrieve_context(state: AgentState) -> AgentState:
    results = qdrant_search(state['question'])
    context = "\n\n".join([results.points[i].payload['text'] for i in range(min(5, len(results.points)))])
    state['context'] = context
    state['next'] = "generate"
    return state

def generator(state: AgentState) -> AgentState:
    history_str = "\n".join([f"Human: {m['human']}\nAI: {m['ai']}" for m in state["chat_history"]])
    llm = llms.get(state["model"], llms["mistral"])
    chain = prompt | llm
    res = chain.invoke({
        "question": state["question"],
        "context": state.get("context", ""),
        "chat_history": history_str
    })
    state["answer"] = res
    state["chat_history"].append({"human": state["question"], "ai": res})
    state["chat_history"] = state["chat_history"][-5:]  # trim history
    state["next"] = "end"
    return state

def validate_output(state: AgentState) -> AgentState:
    answer = state["answer"]
    question = state["question"]

    # Basic validation logic - you can replace this with custom rules or a separate LLM
    if not answer or len(answer.strip()) < 10:
        # If output too short or missing, revise
        revised_answer = f"Sorry, I didn't understand that well. Could you clarify or rephrase?"
        state["answer"] = revised_answer
    elif any(w in answer.lower() for w in ["i don't know", "no idea", "cannot help"]):
        # If unsure response, revise
        revised_answer = f"Hmm, I'm not sure about that. Let's try again. Could you explain more about your question?"
        state["answer"] = revised_answer

    # Otherwise, assume answer is fine
    state["next"] = "end"
    return state

# ------------------- Graph Builder -------------------
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("retriever", retrieve_context)
    builder.add_node("generator", generator)
    builder.add_node("validator", validate_output)
    builder.set_entry_point("retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", "validator")
    builder.add_edge("validator", END)
    return builder.compile()


# ------------------- Streamlit UI -------------------
def main():
    st.set_page_config(page_title="CompanionAI Chatbot", page_icon="🤖")
    st.title("Chat with CompanionAI 🤖")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    model = st.sidebar.selectbox("Choose a model", ["gemma2", "mistral"])
    if st.sidebar.button("New Chat"):
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message("human"):
            st.write(msg["human"])
        with st.chat_message("ai"):
            st.write(msg["ai"])

    user_input = st.chat_input("Type your message here...")
    if user_input:
        with st.chat_message("human"):
            st.write(user_input)
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                graph = build_graph()
                inputs = {
                    "question": user_input,
                    "chat_history": st.session_state.chat_history,
                    "model": model,
                    "context": None,
                    "answer": None,
                    "next": "retrieve"
                }
                final_state = graph.invoke(inputs)
                st.session_state.chat_history = final_state['chat_history']
                st.write(final_state['answer'])

if __name__ == "__main__":
    main()