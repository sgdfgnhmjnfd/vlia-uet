"""Tiền/hậu xử lý ảnh và action chunk cho VLIA-SmolVLA.

Phase 3 additionally uses the following prompt template:
    <think>{think}</think><intention>{intention}</intention><action>
"""

PROMPT_TEMPLATE = "<think>{think}</think><intention>{intention}</intention><action>"


def build_prompt(think: str, intention: str) -> str:
    return PROMPT_TEMPLATE.format(think=think, intention=intention)
