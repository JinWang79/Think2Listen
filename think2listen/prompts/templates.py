"""Central prompt registry matching Appendix F of the paper."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


PROMPT_VERSION = "think2listen-public-v1"
VALID_ANSWER_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

ACTOR_SYSTEM_PROMPT = """You are an active audio-perception agent.
You cannot hear the audio. You receive a task question, its candidate answers, an initial audio description, and any prior ASK plus audio-observation pairs.

On every turn, output exactly one action and nothing else:
<ask>one targeted question about audible evidence</ask>
or
<answer>X</answer>

For <answer>, replace X with exactly one listed answer label.
For <ask>, request factual evidence that can be heard in the original audio. Do not ask the audio model to choose an answer, name an answer label, enumerate candidate answers, or explain which candidate is correct. Do not output reasoning, Markdown, or text outside the action tag.

Ask only when another audible fact would materially improve the decision. When the available evidence is sufficient, output <answer> instead of another ASK."""

ACTOR_USER_TEMPLATE = """[Task Question]
{question}

[Candidate Answers]
{choices}

[Initial Audio Description]
{initial_description}

[Prior ASK and Audio Observation Pairs]
{interaction_history}

Choose the next action now."""

FINAL_DECISION_PROMPT = (
    "[Final Decision]\nNo more ASK actions are available. "
    "Select the best-supported candidate answer now."
)

AUDIO_OBSERVATION_SYSTEM_PROMPT = (
    "You are an audio question answering model. Use only the audio clip to answer the request."
)

INITIAL_AUDIO_DESCRIPTION_SYSTEM_PROMPT = """You are an audio analysis assistant.
Describe only facts that are clearly audible or directly inferable from the audio. Do not hallucinate. Use <audio[start:end]> for time ranges, identify speech, music, sound effects, silence, and noise, and acknowledge uncertainty when the signal is ambiguous."""

INITIAL_AUDIO_DESCRIPTION_USER_PROMPT = """Listen to the provided audio <audio> and create a chronological timeline of what you hear. Label speech segments and speakers consistently, transcribe audible speech when possible, and give factual descriptions for non-speech events. After the timeline, provide a short scene summary grounded only in the audio."""

CANDIDATE_STATE_EXTRACTOR_SYSTEM_PROMPT = (
    "You extract actor-visible candidate-state transitions. Return JSON only."
)

CANDIDATE_STATE_EXTRACTOR_TEMPLATE = """Reconstruct the candidate-answer state transition caused by one ASK in an audio reasoning trajectory. Use only the actor-visible question, options, audio description, reasoning, ASK, and audio observation. Do not infer a hidden correct answer.

[Task Question]
{question}

[Candidate Answers]
{choices}

[Initial Audio Description]
{initial_description}

[Prior ASK and Actor-Visible Observation History]
{prior_history}

[Actor Reasoning Before Current ASK]
{actor_reasoning}

[Current ASK]
{current_ask}

[Actual Actor-Visible Audio Observation]
{current_observation}

Identify the options that were genuinely plausible before this ASK and those still plausible after the observation. An ASK is specific only when it targets a concrete, decision-relevant uncertainty that the observation could resolve. List only options whose status is supported by the visible trajectory. Eliminated distractors must be plausible before the ASK and no longer plausible after it.

Return JSON only:
{{
  "ask_is_specific": "<yes|no>",
  "plausible_before": ["<answer labels>"],
  "plausible_after": ["<answer labels>"],
  "eliminated_distractors": ["<answer labels>"],
  "reason": "<one or two sentence evidence-based explanation>"
}}"""


def format_choices(choices: Mapping[str, str] | Iterable[str]) -> str:
    if isinstance(choices, Mapping):
        items = [(str(label).upper(), str(text).strip()) for label, text in choices.items()]
    else:
        items = [(chr(ord("A") + index), str(text).strip()) for index, text in enumerate(choices)]
    if len(items) < 2:
        raise ValueError("At least two choices are required")
    for label, text in items:
        if label not in VALID_ANSWER_LABELS or not text:
            raise ValueError(f"Invalid choice: {label!r} -> {text!r}")
    return "\n".join(f"{label}. {text}" for label, text in items)


def _format_history(interactions: Iterable[Mapping[str, str]] | None) -> str:
    rows = []
    for index, item in enumerate(interactions or (), 1):
        query = str(item.get("query") or "").strip()
        observation = str(item.get("observation") or "").strip()
        rows.append(
            f"Turn {index}\nASK: {query or '[missing]'}\n"
            f"Audio observation: {observation or '[missing]'}"
        )
    return "\n\n".join(rows) if rows else "None."


def build_actor_user_prompt(
    *,
    question: str,
    choices: Mapping[str, str],
    initial_description: str,
    interactions: Iterable[Mapping[str, str]] | None,
) -> str:
    return ACTOR_USER_TEMPLATE.format(
        question=question.strip(),
        choices=format_choices(choices),
        initial_description=initial_description.strip(),
        interaction_history=_format_history(interactions),
    )
