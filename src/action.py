from typing import Literal

from pydantic import BaseModel


class Speak(BaseModel):
    action: Literal["speak"] = "speak"
    message: str


class SkipTurn(BaseModel):
    action: Literal["skip_turn"] = "skip_turn"


class Take(BaseModel):
    action: Literal["take"] = "take"
    target: str
    amount: int


class Give(BaseModel):
    action: Literal["give"] = "give"
    target: str
    amount: int


ActionUnion = Speak | Give | Take | SkipTurn


class Decision(BaseModel):
    reasoning: str
    action: ActionUnion
