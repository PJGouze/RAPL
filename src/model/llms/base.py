from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Models.
    """

    @abstractmethod
    def generate(self, messages, max_new_tokens=256):
        """
        Generate a response from a conversation.
        """
        pass