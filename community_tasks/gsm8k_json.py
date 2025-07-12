from pydantic import BaseModel, Field
from typing import Union

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
import lighteval.tasks.default_prompts as prompt


# 1. Define the JSON output schema using Pydantic
class Gsm8kJson(BaseModel):
    """A structured JSON output for GSM8K."""

    reasoning: str = Field(description="The step-by-step reasoning to solve the math problem.")
    answer: Union[int, float] = Field(description="The final numerical answer.")


# 2. Define a new prompt function to instruct the model to output JSON
def gsm8k_json_prompt(line: dict, task_name: str) -> Doc:
    """
    Defines the prompt for the gsm8k-json task.
    It instructs the model to output a JSON object with 'reasoning' and 'answer' keys.
    """
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n\nFirst, reason step-by-step about the problem. Then, provide your final answer in a JSON object with the keys 'reasoning' and 'answer'.",
        # The original metric expects the gold answer string to be in choices.
        # It will extract the final number from this string.
        choices=[line["answer"]],
        gold_index=0,
        # instruction="",
    )


# 3. Define the new LightevalTaskConfig
gsm8k_json_task = LightevalTaskConfig(
    name="gsm8k-json",
    suite=["community"],
    prompt_function=prompt.gsm8k,
    hf_repo="gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=1024,
    metric=[
        Metrics.expr_gold_metric,
    ],
    stop_sequence=["Question:"],
    trust_dataset=True,
    version=0,
    guided_decoding={"json": Gsm8kJson.model_json_schema()},
)

# 4. Register the new task so it can be used by LightEval
TASKS_TABLE = [gsm8k_json_task]
