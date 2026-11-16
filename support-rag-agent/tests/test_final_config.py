from support_agent.agent import AgentConfig
from support_agent.final_config import FINAL_CONFIG


def test_final_config_is_frozen_to_the_documented_values():
    assert FINAL_CONFIG == AgentConfig(
        retrieval="hybrid_rerank", top_k=5, defense="full", max_steps=4
    )
