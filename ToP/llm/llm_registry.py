from class_registry import ClassRegistry
from abc import ABC, abstractmethod
from typing import List, Optional, Union

class LLM(ABC):
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TEMPERATURE = 0.7

    @abstractmethod
    def generate_response(self,
                          messages: List[str],
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None
                          ) -> Union[List[str], str]:
        pass

    @abstractmethod
    async def generate_response_async(self,
                                      messages: List[str],
                                      max_tokens: Optional[int] = None,
                                      temperature: Optional[float] = None
                                      ) -> Union[List[str], str]:
        pass


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        model = cls.registry.get('QwenChat', model_name)
        return model

