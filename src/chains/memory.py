from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class ChatSessionMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: List[BaseMessage] = []

    def add_user_message(self, message: str):
        self.history.append(HumanMessage(content=message))
        self._trim_history()

    def add_ai_message(self, message: str):
        self.history.append(AIMessage(content=message))
        self._trim_history()

    def get_messages(self) -> List[BaseMessage]:
        return self.history

    def clear(self):
        self.history.clear()

    def _trim_history(self):
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]