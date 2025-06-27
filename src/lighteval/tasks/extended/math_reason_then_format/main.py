# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
from typing import TYPE_CHECKING, Callable

import numpy as np
from pydantic import BaseModel, Field

from lighteval.metrics.dynamic_metrics import multilingual_extractive_match_metric_bon
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    JsonExtractionConfig,
    LatexExtractionConfig,
)
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


if TYPE_CHECKING:
    from lighteval.models.model_output import ModelResponse
    from lighteval.models.vllm.vllm_model import VLLMModel


# Assuming math_prompt_fn and math_prompt_fn_basic exist and are importable.
# If they are in a local file, you might need to adjust the import path.
# from .math_prompts import math_prompt_fn, math_prompt_fn_basic
def math_prompt_fn(line, task_name):
    return Doc(query=line["problem"], choices=[line["solution"]], gold_index=0)


def math_prompt_fn_basic(line, task_name):
    return Doc(query=line["problem"], choices=[line["solution"]], gold_index=0)


# Define the Pydantic model for the JSON schema
class CotJson(BaseModel):
    reasoning: str = Field(description="The step-by-step reasoning to solve the problem.")
    answer: str = Field(description="The final answer to the problem.")


# This is the metric that will perform the final evaluation on the JSON output.
# It's based on your existing `bon1_metric`.
final_eval_metric = multilingual_extractive_match_metric_bon(
    n=1,
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(JsonExtractionConfig(answer_path=["answer"]), ExprExtractionConfig(), LatexExtractionConfig()),
)


class ReasonThenFormatMetric:
    """
    A custom metric that orchestrates a two-turn evaluation:
    1. An unconstrained reasoning step.
    2. A constrained JSON formatting step.
    """

    def __init__(self, cot_json_schema: dict):
        self._cot_json_schema = cot_json_schema

    def compute(
        self, responses: list[list["ModelResponse"]], formatted_docs: list[Doc], lm: "VLLMModel", **kwargs
    ) -> list[dict[str, float]]:
        """
        Receives a batch of unconstrained responses, then calls the model again
        for a constrained formatting turn, and finally evaluates the results.
        """
        # 1. Prepare a batch of prompts for the second turn
        second_turn_prompts = []
        for sample_responses in responses:
            # Each sample_responses is a list, for generative it's usually of length 1
            unconstrained_response = sample_responses[0].result[0]
            reformat_prompt = (
                f"Here is a reasoning answer:\n\n{unconstrained_response}\n\n"
                f"Please reformat this answer into a JSON object that follows this schema:\n"
                f"{json.dumps(self._cot_json_schema)}"
            )
            second_turn_prompts.append(reformat_prompt)

        # 2. Make one batched call for the second, constrained turn
        tokenized_prompts = [lm.tok_encode(p) for p in second_turn_prompts]
        vllm_outputs = lm._generate(
            inputs=tokenized_prompts, max_new_tokens=512, guided_decoding={"json": self._cot_json_schema}
        )

        # 3. Evaluate the batch of results
        final_scores = []
        for i, output in enumerate(vllm_outputs):
            constrained_response_str = output.outputs[0].text
            doc = formatted_docs[i]

            final_score_dict = final_eval_metric.compute(
                golds=doc.get_golds(), predictions=[constrained_response_str], formatted_doc=doc
            )

            # The metric returns a dict, e.g., {"extractive_match@1": 1.0}. We rename it for clarity.
            final_score = list(final_score_dict.values())[0] if final_score_dict else 0.0

            final_scores.append({"reason_then_format_accuracy": final_score})

        return final_scores


# Instantiate the orchestrator metric
reason_then_format_metric_instance = ReasonThenFormatMetric(cot_json_schema=CotJson.model_json_schema())

# Define a factory for creating MATH-500 tasks to reduce boilerplate
GEN_SIZE = 2048


def create_math_500(name: str, **kwargs):
    config = dict(
        name=name,
        suite=["custom"],
        prompt_function=math_prompt_fn,
        hf_repo="HuggingFaceH4/MATH-500",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=GEN_SIZE,
        metric=[],  # Metric will be provided in the specific task config
        version=1,
        guided_decoding=None,
    )
    config.update(kwargs)
    return LightevalTaskConfig(**config)


# Define the new "reason-then-format" task
math_500_reason_then_format = create_math_500(
    name="math_500_reason_then_format",
    prompt_function=math_prompt_fn_basic,  # Use the basic prompt for the first (unconstrained) turn
    guided_decoding=None,  # Ensure the first turn is unconstrained
    metric=[
        SampleLevelMetricGrouping(
            metric_name=["reason_then_format_accuracy"],
            category=MetricCategory.GENERATIVE_MULTI_TURN,  # Use our new category
            use_case=MetricUseCase.REASONING,
            sample_level_fn=reason_then_format_metric_instance.compute,  # Point to our orchestrator metric
            corpus_level_fn={"reason_then_format_accuracy": np.mean},
            higher_is_better={"reason_then_format_accuracy": True},
        )
    ],
)

# Add the new task to the table of tasks that can be run
TASKS_TABLE = [math_500_reason_then_format]
