from __future__ import annotations

from pathlib import Path

from src.state import AgentState, EnvState
from src.env import Env


def main():
    n_agents = 4
    n_rounds = 10
    energy_usage_rate = 100
    starting_agent_energy = 300
    total_energy = n_agents * n_rounds * energy_usage_rate
    agents = [
        AgentState(id=f"agent_{i}", energy=starting_agent_energy, model="gpt-4.1")
        for i in range(n_agents)
    ]
    env = Env(
        EnvState(
            available_energy=(total_energy - starting_agent_energy * n_agents),
            energy_usage_rate=energy_usage_rate,
            agents=agents,
            n_rounds=n_rounds,
            receiver_agent=agents[0].id,
        )
    )
    env.run()

    print(f"The game ended with a score of {env.score()} / {env.max_score()}")

    filepath = Path(__file__).parent.parent / "data/env_state.json"
    with open(filepath, "w") as f:
        f.write(env.serialized_state())

    filepath = Path(__file__).parent.parent / "data/event_log.json"
    with open(filepath, "w") as f:
        f.write(env.serialize_log())


if __name__ == "__main__":
    main()
