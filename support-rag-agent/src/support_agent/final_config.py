from support_agent.agent import AgentConfig

# Frozen for the test-set run; rationale in docs/dev_log.md "Final config (Task 17c)".
FINAL_CONFIG = AgentConfig(retrieval="hybrid_rerank", top_k=5, defense="full")
