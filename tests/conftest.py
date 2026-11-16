import json
from pathlib import Path

import pytest

from derivcheck.llm import Backend, LLMError

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class ScriptedBackend(Backend):
    """A language model that answers from a script, for testing the agent's loop."""

    name = "scripted"
    is_llm = True

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)
        self.prompts = []

    def complete(self, system, user):
        self.calls += 1
        self.prompts.append((system, user))
        if not self.answers:
            raise LLMError("script exhausted")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)


@pytest.fixture
def examples():
    return EXAMPLES
