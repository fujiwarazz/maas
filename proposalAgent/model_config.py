import os

TONGYI_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("PROPOSALS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/peelsannaw/Desktop/codes/maas/mas4proposal/proposalAgent/data",
    # LLM settings
    "llm_provider": "tongyi",
    # "api_key": os.getenv("DASHSCOPE_API_KEY"),
    "api_key": "sk-9ce983386aa74c8f8131eb8ecbf90f58",
    "deep_think_llm": "qwen-plus",
    "quick_think_llm": "qwen-plus",
    "backend_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # Debate and discussion settings
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 3,
    "max_recur_limit": 3,
    # Tool settings
    # "tools": [
    #     "python_repl",
    #     "terminal",
    #     "wikipedia",
    #     "python_code_interpreter",
    #     "python_code_interpreter_sandbox",
    # ],
}
