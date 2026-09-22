"""The state/questions request dialect, lowered onto the context/options core.

A caller describes one record the way the Choice primitive does: a ``state`` plus
named ``questions``, each with ``instructions`` and a ``criteria`` map of option
name to description. The scorer itself still only ever sees a context string and
a list of text options, so this module lowers a request into one ``ChoiceExample``
per question and lifts the scores back into named answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from . import __version__
from .data import ChoiceExample
from .train import move

CHOICE = "choice"
MODEL = f"jevlike-{__version__}"
MAX_OPTIONS = 255


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: dict[str, str | None]


@dataclass(frozen=True)
class ChoiceRequest:
    state: str
    questions: dict[str, ChoiceQuestion]


def _render(value: Any) -> str:
    """Flatten a string, object or array of instructions into one block of text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_render(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_render(item)}" for key, item in value.items())
    raise ValueError(f"instructions must be text, an object or an array, not {type(value).__name__}")


def _parse_question(name: str, payload: Any) -> ChoiceQuestion:
    if not isinstance(payload, dict):
        raise ValueError(f"question {name!r} must be an object")
    kind = payload.get("type", CHOICE)
    if kind != CHOICE:
        raise ValueError(f"question {name!r} has unsupported type {kind!r}")
    if "instructions" not in payload:
        raise ValueError(f"question {name!r} needs instructions")
    criteria = payload.get("criteria")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise ValueError(f"question {name!r} needs at least two criteria")
    if len(criteria) > MAX_OPTIONS:
        raise ValueError(f"question {name!r} has more than {MAX_OPTIONS} criteria")
    descriptions = {}
    for option, description in criteria.items():
        if not isinstance(option, str) or not option:
            raise ValueError(f"question {name!r} needs non-empty criteria names")
        if description is None or description == "":
            descriptions[option] = None
        elif isinstance(description, (str, dict, list)):
            descriptions[option] = _render(description)
        else:
            raise ValueError(f"criteria {option!r} in question {name!r} must be text, an object, an array or null")
    return ChoiceQuestion(_render(payload["instructions"]), descriptions)


def parse_request(payload: Any) -> ChoiceRequest:
    """Validate one request object and return it with criteria names preserved in order."""
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    state = payload.get("state")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("request needs a non-empty string state")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("request needs at least one question")
    return ChoiceRequest(
        state, {name: _parse_question(name, item) for name, item in questions.items()}
    )


def context_for(state: str, question: ChoiceQuestion) -> str:
    """The context the core scorer sees: the state, then that question's instructions."""
    return f"{state}\n\n{question.instructions}"


def options_for(question: ChoiceQuestion) -> tuple[str, ...]:
    """The option texts the core scorer sees, one per criteria name.

    Descriptions ride along with the name so both reach the scorer. They still
    share the checkpoint's option token budget, so long descriptions can be cut.
    """
    return tuple(
        name if description is None else f"{name}: {description}"
        for name, description in question.criteria.items()
    )


def build_answer(names: list[str], probabilities: list[float]) -> dict[str, Any]:
    """Turn one row of option probabilities into a named choice answer.

    Confidence is the margin between the best and second best option, so a sharp
    single peak gives 1.0 and a flat spread gives 0.0.
    """
    values = [float(probability) for probability in probabilities[:len(names)]]
    ranking = sorted(values, reverse=True)
    margin = ranking[0] - ranking[1] if len(ranking) > 1 else ranking[0]
    return {
        "type": CHOICE,
        "choice": names[values.index(max(values))],
        "confidence": round(max(0.0, margin), 6),
        "probabilities": {
            name: round(value, 6) for name, value in zip(names, values)
        },
    }


def build_usage(batch: dict[str, torch.Tensor]) -> dict[str, int]:
    """Count context and option tokens in, one scored option out."""
    return {
        "input_tokens": int(batch["context_mask"].sum() + batch["option_token_mask"].sum()),
        "output_tokens": int(batch["option_mask"].sum()),
    }


@torch.no_grad()
def answer_request(request: ChoiceRequest, model, collator, device) -> dict[str, Any]:
    """Score every question against the one state and return the response object."""
    names = list(request.questions)
    batch = move(collator([
        ChoiceExample(
            context_for(request.state, request.questions[name]),
            options_for(request.questions[name]),
            0,
        )
        for name in names
    ]), device)
    model.eval()
    probabilities = model(batch).softmax(-1).cpu()
    answers = {}
    for row, name in enumerate(names):
        question = request.questions[name]
        answers[name] = build_answer(
            list(question.criteria), probabilities[row, :len(question.criteria)].tolist()
        )
    return {"model": MODEL, "answers": answers, "usage": build_usage(batch)}
