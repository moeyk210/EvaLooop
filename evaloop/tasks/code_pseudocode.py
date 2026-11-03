"""Code -> pseudocode task: the intermediate step of the pseudocode loop.

The standard EvaLoop cycle is code -> natural-language summary -> code. This
task replaces the summary with language-agnostic pseudocode, so the loop
becomes code -> pseudocode -> code. Everything else (generation prompt,
testing, assert-line convention) stays identical, so differences in loop
survival can be attributed to the intermediate representation.
"""

import re
from typing import Any, Dict

from .base import BaseTask
from .code_summarization import wrap_with_assert_line
from .factory import register_task

DEFAULT_PSEUDOCODE_PROMPT = (
    "Write language-agnostic pseudocode for the following code:\n\n"
    "```\n{code}\n```\n\n"
    "Use structured pseudocode keywords (e.g. FUNCTION, IF, ELSE, FOR, WHILE, RETURN) "
    "and plain-English operations. Do not use Python syntax or Python built-in "
    "function names.\n\n"
    "Return only the pseudocode, without explanations or comments."
)

# Matches a response that is entirely one fenced block, e.g. ```text ... ```
_FENCED = re.compile(r"^```[\w+-]*[ \t]*\n(.*?)\n?```$", re.DOTALL)


@register_task("code_pseudocode")
class CodePseudocodeTask(BaseTask):
    """Translate code into pseudocode that seeds the next generation step."""

    def create_prompt(self, input_text: str) -> str:
        template = self.task_config.get("prompt_template", DEFAULT_PSEUDOCODE_PROMPT)
        return template.format(code=input_text)

    def extract_output(self, raw_response: str) -> str:
        """Take the whole response, unwrapping it if it is a single fenced block."""
        text = raw_response.strip()
        match = _FENCED.match(text)
        if match:
            text = match.group(1).strip()
        # A literal triple quote would terminate the docstring wrapper early.
        return text.replace('"""', "'''")

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        """Wrap the pseudocode as a docstring and re-attach the assert line.

        Same convention as the standard loop (see
        CodeSummarizationTask.postprocess), so the function name/signature is
        preserved and the two loop types are directly comparable.
        """
        return wrap_with_assert_line(output, context)
