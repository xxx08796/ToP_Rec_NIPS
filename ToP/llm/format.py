import dataclasses
from typing import List, Literal

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class LLMMessage:
    role: MessageRole
    content: str