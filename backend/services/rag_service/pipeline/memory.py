from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# RAM memory
chat_history_store = ChatMessageHistory()

def get_chat_history():
    history_text = ""

    for msg in chat_history_store.messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    return history_text.strip()


def save_interaction(question: str, answer: str):
    chat_history_store.add_message(HumanMessage(content=question))
    chat_history_store.add_message(AIMessage(content=answer))