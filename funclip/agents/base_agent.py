# Use ABC to write a base class for all agents with enter, close, and __call__ methods
from abc import ABC, abstractmethod

from typing import Any


class BaseAgent(ABC):

    def __init__(self, *args, **kwargs):
        self._is_open = False
        self._is_closed = False

    def __enter__(self):
        if not self._is_open:
            self.open()
            self._is_open = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._is_closed:
            self.close()
            self._is_closed = True

    @abstractmethod
    def open(self) -> None:
        """Open the agent."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the agent."""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Call the agent."""
        pass