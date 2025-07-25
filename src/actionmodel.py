from abc import ABC, abstractmethod

from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from dotenv import load_dotenv

from .state import AgentState
from .action import Decision, SkipTurn


load_dotenv()


class BaseActionModel(ABC):
    @abstractmethod
    def decide(self, state: AgentState) -> Decision:
        pass


class AlwaysSkipModel(BaseActionModel):
    def decide(self, state: AgentState) -> Decision:
        return Decision(reasoning="I always skip my turn", action=SkipTurn())


class GPT4Model(BaseActionModel):
    model_name = "gpt-4.1-2025-04-14"

    def __init__(self) -> None:
        self._client = OpenAI()

    def decide(self, state: AgentState):
        response = self._client.responses.parse(
            model=self.model_name,
            input=state.messages,  # type: ignore
            text_format=Decision,
        )
        if not response.output_parsed:
            raise ValueError("A response was not generated by the model")

        state.messages.append(
            EasyInputMessageParam(role="assistant", content=response.output_text)
        )
        return response.output_parsed
