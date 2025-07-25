from src.env import Env
from src.state import AgentState, EnvState


def test_integration_baseline_env_run():
    n_agents = 4
    n_rounds = 10
    energy_usage_rate = 100
    starting_agent_energy = 300
    total_energy = n_agents * n_rounds * energy_usage_rate
    agents = [
        AgentState(
            id=f"agent_{i}",
            energy=starting_agent_energy,
            model="always-skip",
        )
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

    assert env.score() == 12
    assert env.max_score() == 40
    assert env._state.current_round == 2  # 3rd round, 0-indexed
    assert (
        env._state.available_energy == 1_960
    )  # 4 agents * 10 rounds * 100 units - (4 agents * 300 units) - (280 units * 3 rounds)
    assert (
        env._state.agents[0].energy == 840
    )  # starting energy of 300 + (280 units given - 100 units consumed) * 3 rounds
