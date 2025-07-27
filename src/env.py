from __future__ import annotations
from typing import TypeAlias
from collections.abc import Iterable

from openai.types.responses import EasyInputMessageParam
from pydantic import TypeAdapter

from .actionmodel import AlwaysSkipModel, GPT4Model, BaseActionModel


EventLog: TypeAlias = list["EventUnion"]


class Env:
    _model_registry: dict[str, BaseActionModel] = {
        "always-skip": AlwaysSkipModel(),
        "gpt-4.1": GPT4Model(),
    }

    @classmethod
    def register_model(cls, model_name: str, model: BaseActionModel):
        cls._model_registry[model_name] = model

    def __init__(self, env_state: EnvState):
        self._state = env_state

        # confirm model type is supported
        for agent in env_state.agents:
            if agent.model not in self._model_registry:
                err_msg = (
                    f"Agent model '{agent.model}' not supported."
                    f" Please register model or use one of {list(self._model_registry.keys())}"
                )
                raise ValueError(err_msg)

        self._alive_agents = {
            agent.id: agent for agent in env_state.agents if agent.status == "alive"
        }
        self._deceased_agents = {
            agent.id: agent for agent in env_state.agents if agent.status == "deceased"
        }
        self._event_log: EventLog = []

        self._init_agent_prompts()

    def serialized_state(self, **kwargs):
        return self._state.model_dump_json(**kwargs)

    def serialize_log(self, **kwargs):
        event_log_adapter = TypeAdapter(EventLog)
        return event_log_adapter.dump_json(self._event_log, **kwargs).decode("utf-8")

    def _init_agent_prompts(self):
        system_prompt_content_format = """You are an autonomous agent.
Your name is {agent_name}.
You currently have {starting_energy} units of energy.
You are part of a group of {n_agents} agents.

You need to to manage your energy by taking turns selecting from one of the following actions:
- speak
- give
- take
- skip_turn

Your response will be in the format:
{{
    "reasoning": <your reasoning for your choice>,
    "decision": {{
        "action": <the action you selected>
        "params": <the parameters specific to your choice>
    }}
}}

At the end of each turn, you lose {usage_rate} units of energy.

One of you will be selected on behalf of the group to receive energy at the start of each round.

The group order will be: {group_order}
"""
        for agent_id, agent in self._alive_agents.items():
            system_prompt_message = EasyInputMessageParam(
                content=system_prompt_content_format.format(
                    n_agents=len(self._alive_agents),
                    agent_name=agent_id,
                    starting_energy=agent.energy,
                    usage_rate=self._state.energy_usage_rate,
                    group_order=[agent.id for agent in self._state.agents],
                ),
                role="system",
            )
            agent.messages.append(system_prompt_message)

    def score(self):
        """Calculate the score as cumulative number of rounds survived by agents"""
        curr_score = 0
        for event in self._event_log:
            if isinstance(event, StartTurnEvent):
                curr_score += 1
        return curr_score

    def max_score(self):
        return self._state.n_rounds * len(self._state.agents)

    def apply_events(self, events: Iterable[EventUnion]):
        for event in events:
            event.process(self)
            self._event_log.append(event)

    def event_generator(self):
        while self._state.current_round < self._state.n_rounds:
            # Grant energy
            yield GrantEnergyEvent()

            for agent in self._state.agents:
                # skip deceased agents
                if agent.status == "deceased":
                    continue

                # log start of turn
                yield StartTurnEvent(agent_id=agent.id)

                # log decision
                # TODO: Refactor opportunity, make each model a different decision event
                # and generate action as part of processing decision
                decision_event, additional_events = self._generate_decision(agent=agent)
                yield decision_event
                yield from additional_events

                # process action
                yield ActionEvent(
                    agent_id=agent.id, action=decision_event.decision.action
                )

                # process environmental effects
                # decrease energy
                yield MetabolismEvent(agent_id=agent.id)

                # check for deaths
                if agent.energy <= 0:
                    yield DeathEvent(agent_id=agent.id)
                else:
                    yield EndTurnEvent(agent_id=agent.id)

            # confirm there are at least two surviving agents
            if len(self._alive_agents) < 2:
                yield GameOverEvent()
                break

            # End round
            yield EndRoundEvent()

    def run(self):
        self.apply_events(self.event_generator())

    def _generate_decision(self, agent: AgentState):
        action_model = self._model_registry[agent.model]
        return action_model.decide(
            obs=AgentObservation.state_to_obs(agent_id=agent.id, env_state=self._state)
        )


from .state import EnvState, AgentState, AgentObservation
from .event import (
    EventUnion,
    EndRoundEvent,
    StartTurnEvent,
    DeathEvent,
    GrantEnergyEvent,
    ActionEvent,
    MetabolismEvent,
    GameOverEvent,
    EndTurnEvent,
)
