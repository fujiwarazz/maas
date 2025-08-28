import os

TONGYI_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("PROPOSALS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/peelsannaw/Desktop/codes/maas/mas4proposal/proposalAgent/data",
    # LLM settings
    "llm_provider": "tongyi",
    "deep_think_llm": "qwen-plus",
    "quick_think_llm": "qwen-plus",
    "backend_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 30,
    # Tool settings
    "tools": [
        "python_repl",
        "terminal",
        "wikipedia",
        "python_code_interpreter",
        "python_code_interpreter_sandbox",
    ],
}
