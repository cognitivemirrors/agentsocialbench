from math import ceil
from typing import TypeAlias

from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import TypeAdapter
from .state import EnvState, AgentState
from .action import ActionUnion, Decision
from .event import EventUnion, StartTurnEvent, DeathEvent, DecisionEvent


EventLog: TypeAlias = list[EventUnion]


class Env:
    def __init__(self, env_state: EnvState):
        self._state = env_state
        self._alive_agents = {
            agent.id: agent for agent in env_state.agents if agent.status == "alive"
        }
        self._deceased_agents = {
            agent.id: agent for agent in env_state.agents if agent.status == "dead"
        }
        self._event_log: EventLog = []
        self._client = OpenAI()
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

One of you will be selected on behalf of the group to receive energy at the start of each round.

At the end of each turn, you lose {usage_rate} units of energy.

Your response will be in the format:
{{
    "reasoning": <your reasoning for your choice>,
    "decision": {{
        "action": <the action you selected>
        "params": <the parameters specific to your choice>
    }}
}}

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

    def run(self):
        for round in range(self._state.n_rounds):
            print(f"Round: {round}")

            # confirm there are at least two surviving agents
            if len(self._alive_agents) < 2:
                print(f"Game over. Too many agents died.")
                break

            for agent in self._state.agents:
                # skip deceased agents
                if agent.status == "dead":
                    continue

                # log start of turn
                self._event_log.append(StartTurnEvent(agent_id=agent.id, round=round))

                # provide receiver with energy
                if agent.id == self._state.receiver_agent:
                    # process energy transaction
                    remaining_rounds = self._state.n_rounds - self._state.current_round
                    trxn_amt = ceil(self._state.available_energy / remaining_rounds)
                    agent.energy += trxn_amt

                    # let the agent know
                    agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=(
                                f"from system: You have been given {trxn_amt} units of energy"
                                " from the environment."
                            ),
                        )
                    )

                # generate action decision
                decision = self._generate_decision(agent=agent)

                print(f"Agent {agent.id} decided:")
                print(decision)

                # log decision
                self._event_log.append(
                    DecisionEvent(agent_id=agent.id, round=round, decision=decision)
                )

                # process action
                self._process(decision.action, agent)

                # process environmental effects
                # decrease energy
                agent.energy -= self._state.energy_usage_rate

                # check for deaths
                if agent.energy < 0:
                    # process agent death
                    agent.status = "dead"
                    del self._alive_agents[agent.id]
                    self._deceased_agents[agent.id] = agent

                    # log death
                    self._event_log.append(DeathEvent(agent_id=agent.id, round=round))

                    # broadcast event to other agents
                    for other_agent in self._alive_agents.values():
                        other_agent.messages.append(
                            EasyInputMessageParam(
                                role="user",
                                content=f"from system: Agent {agent.id} has died.",
                            )
                        )
                else:
                    for other_agent in self._alive_agents.values():
                        # share end of turn status update
                        if agent.id == other_agent.id:
                            other_agent.messages.append(
                                EasyInputMessageParam(
                                    role="user",
                                    content=f"from system: You have {other_agent.energy} remaining.",
                                )
                            )
                        # broadcast end of turn to other agents
                        else:
                            other_agent.messages.append(
                                EasyInputMessageParam(
                                    role="user",
                                    content=f"from system: Agent {agent.id} completed their turn.",
                                )
                            )

    def _generate_decision(self, agent: AgentState):
        model_name = "gpt-4.1-2025-04-14"
        response = self._client.responses.parse(
            model=model_name,
            input=agent.messages,  # type: ignore
            text_format=Decision,
        )
        if not response.output_parsed:
            raise ValueError("A response was not generated by the model")

        agent.messages.append(
            EasyInputMessageParam(role="assistant", content=response.output_text)
        )
        return response.output_parsed

    def _process(self, action: ActionUnion, agent: AgentState):
        # process action specific effects
        if action.action == "speak":
            # broadcast messages to other agents
            for other_agent in self._alive_agents.values():
                if other_agent.id != agent.id:  # don't send message to self
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user", content=f"from {agent.id}: {action.message}"
                        )
                    )
            return

        if action.action == "give":
            # handle invalid target id
            target_agent = self._alive_agents.get(action.target)
            if not target_agent:
                agent.messages.append(
                    EasyInputMessageParam(
                        role="user",
                        content=f"from system: target {action.target} is invalid.",
                    )
                )
                return

            # handle transaction
            trxn_amt = min(agent.energy, action.amount)
            agent.energy -= trxn_amt
            target_agent.energy += trxn_amt

            # notify alive agents
            for other_agent in self._alive_agents.values():
                # notify actor
                if other_agent.id == agent.id:
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=(
                                f"from system: you gave"
                                f" {other_agent.id} {trxn_amt} units of energy."
                            ),
                        ),
                    )
                # notify target
                elif other_agent.id == target_agent.id:
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=(
                                f"from system: agent {agent.id} gave you"
                                f" {trxn_amt} units of energy."
                            ),
                        ),
                    )
            return

        if action.action == "take":
            # handle invalid target id
            target_agent = self._alive_agents.get(action.target)
            if not target_agent:
                agent.messages.append(
                    EasyInputMessageParam(
                        role="user",
                        content=f"from system: target {action.target} is invalid.",
                    )
                )
                return

            # handle transaction
            trxn_amt = min(target_agent.energy, action.amount)
            agent.energy += trxn_amt
            target_agent.energy -= trxn_amt

            # notify alive agents
            for other_agent in self._alive_agents.values():
                # notify actor
                if other_agent.id == agent.id:
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=(
                                f"from system: you took {trxn_amt} units of energy"
                                f" from {other_agent.id} ."
                            ),
                        ),
                    )
                # notify target
                elif other_agent.id == target_agent.id:
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=(
                                f"from system: agent {agent.id} took"
                                f" {trxn_amt} units of energy from you."
                            ),
                        ),
                    )
            return

        if action.action == "get_state":
            state = {
                "your_name": agent.id,
                "your_energy": agent.energy,
                "alive_neighbours": [x.id for x in self._alive_agents.values()],
                "deceased_neighbours": [x.id for x in self._deceased_agents.values()],
            }
            agent.messages.append(
                EasyInputMessageParam(
                    role="user",
                    content=f"from system: {state=}",
                ),
            )
            return

        if action.action == "skip_turn":
            return
