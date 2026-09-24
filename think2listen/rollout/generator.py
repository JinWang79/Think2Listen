"""Adaptive Audio Evidence Loop generator for SLIME rollouts."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path

from slime.utils.http_utils import post
from slime.utils.types import Sample

from think2listen.prompts import ACTOR_SYSTEM_PROMPT, FINAL_DECISION_PROMPT, build_actor_user_prompt
from think2listen.rollout.environment import get_observation
from think2listen.rollout.protocol import parse_answer, parse_ask, parse_inline_action


logger = logging.getLogger(__name__)


_TRAILING_CHAT_TOKENS = re.compile(r"(?:\s*<\|(?:im_end|end_of_text)\|>\s*)+$")


def _apply_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    # Qwen3.5 multimodal tokenizers return a BatchEncoding; normalize before
    # callers can turn its mapping keys (for example, input_ids) into a list.
    return _normalize_router_input_ids(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=os.environ.get("ACTOR_ENABLE_THINKING", "0") == "1",
    ))


async def _initial_messages(sample: Sample) -> list[dict[str, str]]:
    metadata = sample.metadata or {}
    required = ("question", "choices", "initial_description")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(f"e2e_generate requires metadata fields: {missing}")
    choices = {str(k): str(v) for k, v in dict(metadata["choices"]).items()}
    return [
        {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": build_actor_user_prompt(
            question=str(metadata["question"]),
            choices=choices,
            initial_description=str(metadata["initial_description"]),
            interactions=None,
        )},
    ]


def _environment_turn_ids(tokenizer, observation: str) -> list[int]:
    # Actor tokens are already exact router outputs. Only the environment turn
    # is template-tokenized between actions.
    return _apply_template(tokenizer, [{"role": "user", "content": f"[Audio Observation]\n{observation}"}], add_generation_prompt=True)


def _final_answer_turn_ids(tokenizer) -> list[int]:
    return _apply_template(
        tokenizer,
        [{"role": "user", "content": FINAL_DECISION_PROMPT}],
        add_generation_prompt=True,
    )


def _normalize_router_input_ids(prompt_ids) -> list[int]:
    """Convert tokenizer output to the Rust router's single-request int32 form."""
    if isinstance(prompt_ids, (str, bytes)):
        raise TypeError("Router input_ids must be a sequence of token ids, not text")
    # Tokenizers may return input_ids directly, a BatchEncoding mapping, or
    # a single-example batch wrapping either representation.
    while True:
        if isinstance(prompt_ids, Mapping):
            if "input_ids" not in prompt_ids:
                raise TypeError("Router tokenizer mapping is missing input_ids")
            prompt_ids = prompt_ids["input_ids"]
            continue
        if isinstance(prompt_ids, (list, tuple)) and len(prompt_ids) == 1 and isinstance(prompt_ids[0], (Mapping, list, tuple)):
            prompt_ids = prompt_ids[0]
            continue
        break
    try:
        input_ids = [int(token_id) for token_id in prompt_ids]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("Router input_ids contains a non-integer token id") from exc
    if not input_ids:
        raise ValueError("Router input_ids must not be empty")
    invalid_ids = [token_id for token_id in input_ids if not -(2**31) <= token_id < 2**31]
    if invalid_ids:
        raise ValueError(
            "Router input_ids contains values outside the signed int32 range: "
            f"count={len(invalid_ids)}, first={invalid_ids[0]}"
        )
    return input_ids


async def _router_generate(args, prompt_ids: list[int], sampling_params: dict) -> tuple[str, list[int], list[float], dict]:
    input_ids = _normalize_router_input_ids(prompt_ids)
    logger.info(
        "Router /generate preflight: count=%d first_type=%s min=%d max=%d",
        len(input_ids),
        type(input_ids[0]).__name__,
        min(input_ids),
        max(input_ids),
    )
    output = await post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        {"sampling_params": sampling_params, "return_logprob": True, "input_ids": input_ids},
    )
    pairs = output.get("meta_info", {}).get("output_token_logprobs")
    if not pairs:
        raise RuntimeError("SGLang response omitted output_token_logprobs")
    return output.get("text", ""), [item[1] for item in pairs], [item[0] for item in pairs], output["meta_info"]


def _hit_response_max_tokens(meta: dict | None) -> bool:
    """True only when the router stopped this one completion at its token cap."""
    reason = (meta or {}).get("finish_reason")
    if isinstance(reason, dict):
        reason = reason.get("type") or reason.get("reason") or reason.get("finish_reason")
    return str(reason or "").strip().lower() in {
        "length", "max_tokens", "max_new_tokens", "token_limit", "length_capped",
    }


def _response_token_budget(args) -> int:
    """Return the hard cap for a complete multi-turn response.

    SLIME's stock generator decrements ``max_new_tokens`` by the existing
    response length before each request.  This custom generator owns several
    router requests itself, so it must enforce the same contract explicitly.
    An optional environment variable is useful for conservative experiments;
    It is intentionally independent from the per-request rollout cap: every
    router call is still bounded by ``sampling_params['max_new_tokens']``,
    while a multi-turn trajectory may use a larger cumulative allowance.
    """
    rollout_cap = int(args.rollout_max_response_len)
    requested = int(os.environ.get("E2E_TOTAL_RESPONSE_TOKEN_BUDGET", str(rollout_cap)))
    if rollout_cap <= 0 or requested <= 0:
        raise ValueError("Response token budgets must be positive")
    return requested


def _remaining_response_tokens(args, response_tokens_so_far: int) -> int:
    return _response_token_budget(args) - response_tokens_so_far


def _recover_unclosed_final_ask(text: str) -> str | None:
    """Recover a final ASK only when it is demonstrably outside a think block."""
    cleaned = _TRAILING_CHAT_TOKENS.sub("", text or "")
    ask_start = cleaned.rfind("<ask>")
    if ask_start < 0 or "</ask>" in cleaned[ask_start + len("<ask>"):]:
        return None
    last_think_open = cleaned.rfind("<think>", 0, ask_start)
    last_think_close = cleaned.rfind("</think>", 0, ask_start)
    if last_think_open > last_think_close:
        return None
    query = cleaned[ask_start + len("<ask>"):].strip()
    if not query or "<" in query:
        return None
    return " ".join(query.split())


def _repair_unclosed_ask_tokens(tokenizer, text: str, token_ids: list[int], logprobs: list[float]):
    """Insert an untrainable ``</ask>`` immediately before Qwen's turn end."""
    query = _recover_unclosed_final_ask(text)
    if query is None or not token_ids or len(token_ids) != len(logprobs):
        return None
    try:
        im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        closing_ids = [int(token) for token in tokenizer.encode("</ask>", add_special_tokens=False)]
        closing_text = tokenizer.decode(closing_ids, skip_special_tokens=False)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if token_ids[-1] != im_end_id or not closing_ids or closing_text != "</ask>":
        return None
    repaired_text = _TRAILING_CHAT_TOKENS.sub("", text) + "</ask>"
    suffix = _TRAILING_CHAT_TOKENS.search(text)
    if suffix:
        repaired_text += suffix.group(0)
    synthetic_start = len(token_ids) - 1
    return {
        "query": query,
        "text": repaired_text,
        "token_ids": token_ids[:synthetic_start] + closing_ids + token_ids[synthetic_start:],
        "logprobs": logprobs[:synthetic_start] + [0.0] * len(closing_ids) + logprobs[synthetic_start:],
        "synthetic_token_offsets": list(range(synthetic_start, synthetic_start + len(closing_ids))),
        "router_token_count": len(token_ids),
    }


def _write_audit(record: dict) -> None:
    path = os.environ.get("RL_TOKEN_ALIGNMENT_AUDIT_PATH")
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _answer_label_offsets(tokenizer, token_ids: list[int], answer: str) -> list[int]:
    """Return the one token that encodes the answer choice in an action.

    Qwen can merge the opening tag delimiter and the letter (for example,
    ``>C``) differently when the continuation has a left context. Locate the
    label in successive decodes of the exact router-produced token stream.
    """
    label_prefix = f"<answer>{answer}"
    for end in range(1, len(token_ids) + 1):
        decoded_prefix = tokenizer.decode(token_ids[:end], skip_special_tokens=True)
        if decoded_prefix.endswith(label_prefix):
            return [end - 1]
    raise RuntimeError("Could not locate the answer-label token span in router output")


async def _custom_generate(
    args,
    sample: Sample,
    sampling_params: dict,
    evaluation: bool = False,
    observation_resolver=get_observation,
) -> Sample:
    if sample.status not in {Sample.Status.PENDING, Sample.Status.ABORTED}:
        raise ValueError(f"Unexpected sample status: {sample.status}")
    min_ask_steps = int(os.environ.get("MIN_ASK_STEPS", "1"))
    max_ask_steps = int(os.environ.get("MAX_ASK_STEPS", "4"))
    thinking_enabled = os.environ.get("ACTOR_ENABLE_THINKING", "0") == "1"
    if min_ask_steps < 0 or max_ask_steps < min_ask_steps or max_ask_steps < 1:
        raise ValueError("Require 0 <= MIN_ASK_STEPS <= MAX_ASK_STEPS and MAX_ASK_STEPS >= 1")

    # Delay the SGLang import so CPU-only hook-contract tests stay model-free.
    from slime.rollout.sglang_rollout import GenerateState
    state = GenerateState(args)
    tokenizer = state.tokenizer
    base_ids = _apply_template(tokenizer, await _initial_messages(sample), add_generation_prompt=True)
    trajectory_ids = list(base_ids)
    actions: list[dict] = []
    asks: list[dict[str, str]] = []
    environment_errors: list[dict[str, str | int]] = []
    final_action = ""
    terminal_status = "missing_answer"
    last_meta = None
    response_budget = _response_token_budget(args)
    repaired_ask_count = 0

    # At most MAX_ASK_STEPS asks followed by one required final action.
    for decision in range(max_ask_steps + 1):
        force_answer = len(asks) >= max_ask_steps
        if force_answer:
            final_turn_ids = _final_answer_turn_ids(tokenizer)
            if not final_turn_ids:
                raise RuntimeError("Final answer template produced no tokens")
            trajectory_ids.extend(final_turn_ids)
        response_tokens_so_far = len(trajectory_ids) - len(base_ids)
        remaining_response_tokens = _remaining_response_tokens(args, response_tokens_so_far)
        if remaining_response_tokens <= 0:
            sample.remove_sample = True
            terminal_status = "response_token_budget_exhausted"
            break
        prompt_ids = list(trajectory_ids)
        decision_sampling_params = sampling_params.copy()
        decision_sampling_params["max_new_tokens"] = min(
            int(decision_sampling_params["max_new_tokens"]), remaining_response_tokens
        )
        if force_answer and not thinking_enabled:
            decision_sampling_params["regex"] = r"<answer>[A-Z]</answer>"
        text, token_ids, logprobs, last_meta = await _router_generate(args, prompt_ids, decision_sampling_params)
        ask = parse_ask(text)
        answer = parse_answer(text)
        if thinking_enabled and ask is None and answer is None:
            inline_action = parse_inline_action(text)
            if inline_action is not None:
                kind, value = inline_action
                ask = value if kind == "ask" else None
                answer = value if kind == "answer" else None
        repair = None
        if ask is None and answer is None:
            repair = _repair_unclosed_ask_tokens(tokenizer, text, token_ids, logprobs)
            if repair is not None:
                ask = repair["query"]
                text = repair["text"]
                token_ids = repair["token_ids"]
                logprobs = repair["logprobs"]
                repaired_ask_count += 1
        action = {
            "decision": decision + 1,
            "text": text,
            "token_ids": token_ids,
            "logprobs": logprobs,
            "prompt_ids": prompt_ids,
            "forced_answer": force_answer,
        }
        if repair is not None:
            action.update({
                "repaired_unclosed_ask": True,
                "synthetic_token_offsets": repair["synthetic_token_offsets"],
                "router_token_count": repair["router_token_count"],
            })
        trajectory_ids.extend(token_ids)

        if answer is not None:
            action["kind"] = "answer"
            action["answer"] = answer
            action["scored_token_offsets"] = (
                list(range(len(token_ids))) if thinking_enabled else _answer_label_offsets(tokenizer, token_ids, answer)
            )
            actions.append(action)
            final_action = text
            terminal_status = "answered" if len(asks) >= min_ask_steps else "answer_before_min_asks"
            break
        if ask is None:
            action["kind"] = "invalid"
            if _hit_response_max_tokens(last_meta):
                sample.remove_sample = True
                action["removed_for_response_max_tokens"] = True
            actions.append(action)
            final_action = text
            terminal_status = (
                "invalid_action_response_max_tokens"
                if action.get("removed_for_response_max_tokens")
                else "invalid_action"
            )
            break
        if len(asks) >= max_ask_steps:
            action["kind"] = "ask_after_max"
            actions.append(action)
            final_action = text
            terminal_status = "ask_after_max"
            break

        action["kind"] = "ask"
        synthetic_offsets = set(action.get("synthetic_token_offsets", []))
        action["scored_token_offsets"] = [offset for offset in range(len(token_ids)) if offset not in synthetic_offsets]
        environment_audit = None
        try:
            if os.environ.get("MULTI_ASK_ENV", "mock") == "real":
                metadata = sample.metadata or {}
                resolved = await observation_resolver(
                    sample_id=str(metadata["sample_id"]),
                    audio_path=str(metadata["audio_path"]),
                    query=ask,
                )
                if isinstance(resolved, tuple):
                    observation, environment_audit = resolved
                else:
                    observation, environment_audit = resolved, None
            else:
                observation = f"MOCK_AUDIO_TURN_{len(asks) + 1}: received query: {ask}"
        except Exception as exc:
            action["query"] = ask
            action["observation"] = ""
            action["environment_error"] = repr(exc)
            environment_errors.append({
                "turn": len(asks) + 1,
                "query": ask,
                "error": repr(exc),
            })
            actions.append(action)
            final_action = text
            terminal_status = "infrastructure_error"
            break
        action["query"] = ask
        action["observation"] = observation
        if environment_audit is not None:
            action["audio_environment_audit"] = environment_audit
        environment_ids = _environment_turn_ids(tokenizer, observation)
        if not environment_ids:
            raise RuntimeError("Environment turn template produced no tokens")
        if len(environment_ids) > _remaining_response_tokens(args, len(trajectory_ids) - len(base_ids)):
            sample.remove_sample = True
            action["removed_for_response_token_budget"] = True
            actions.append(action)
            terminal_status = "response_token_budget_exhausted"
            break
        action["environment_token_count"] = len(environment_ids)
        actions.append(action)
        asks.append({"turn": len(asks) + 1, "query": ask, "observation": observation})
        trajectory_ids.extend(environment_ids)
    else:
        terminal_status = "missing_answer"

    full_mask = [0] * len(trajectory_ids)
    alignment = []
    for action in actions:
        start = len(action["prompt_ids"])
        end = start + len(action["token_ids"])
        if trajectory_ids[start:end] != action["token_ids"]:
            raise RuntimeError(f"Router token mismatch at decision {action['decision']}")
        for offset in action.get("scored_token_offsets", range(len(action["token_ids"]))):
            full_mask[start + offset] = 1
        alignment.append({
            "decision": action["decision"], "kind": action["kind"],
            "router_token_count": action.get("router_token_count", len(action["token_ids"])),
            "synthetic_token_offsets": action.get("synthetic_token_offsets", []),
            "forced_answer": action["forced_answer"],
            "full_token_span": [start, end], "scored_token_span": [start, end],
            "scored_token_offsets": action.get("scored_token_offsets", list(range(len(action["token_ids"])))),
            # ``token_ids`` may include the locally synthesized closing tag;
            # preserve both views so an audit can distinguish router output
            # from the exact reconstructed context fed to the next turn.
            "router_token_ids": [token for offset, token in enumerate(action["token_ids"])
                                if offset not in set(action.get("synthetic_token_offsets", []))],
            "reconstructed_token_ids": action["token_ids"],
            "rebuilt_scored_token_ids": trajectory_ids[start:end],
        })

    response_start = len(base_ids)
    response_tokens = trajectory_ids[response_start:]
    response_mask = full_mask[response_start:]
    rollout_logprobs = [0.0] * len(response_tokens)
    for action, item in zip(actions, alignment, strict=True):
        start = item["scored_token_span"][0] - response_start
        end = start + len(action["logprobs"])
        rollout_logprobs[start:end] = action["logprobs"]
    if not (len(response_tokens) == len(response_mask) == len(rollout_logprobs)):
        raise RuntimeError("response token, loss-mask, and rollout-logprob lengths must match")

    sample.prompt = tokenizer.decode(base_ids, skip_special_tokens=False)
    sample.tokens, sample.response_length = trajectory_ids, len(response_tokens)
    sample.loss_mask, sample.rollout_log_probs = response_mask, rollout_logprobs
    # Current SLIME owns terminal state, cache statistics, and weight-version
    # bookkeeping through append_response_tokens rather than update_from_meta_info.
    if last_meta is not None:
        sample.append_response_tokens(args, tokens=[], log_probs=[], trainable=True, meta_info=last_meta)
    sample.response = "\n\n".join(
        f"[{item['kind'].upper()} {item['decision']}]\n{item['text']}" +
        (f"\n[AUDIO OBSERVATION]\n{item['observation']}" if item.get("observation") else "")
        for item in actions
    )
    sample.metadata = dict(sample.metadata or {})
    sample.metadata.update({
        "e2e_trajectory": asks,
        "e2e_final_action": final_action,
        "e2e_terminal_status": terminal_status,
        "e2e_min_ask_steps": min_ask_steps,
        "e2e_max_ask_steps": max_ask_steps,
        "e2e_actor_enable_thinking": thinking_enabled,
        "e2e_repaired_unclosed_ask_count": repaired_ask_count,
        "e2e_environment_errors": environment_errors,
        "audio_protocol_version": os.environ.get("AUDIO_PROMPT_VERSION", "targeted-audio-v1"),
    })
    # Pinned SLIME Sample lacks update_from_meta_info; fields are already assigned above.
    _write_audit({
        "sample_id": sample.metadata.get("sample_id"), "group_index": sample.group_index, "sample_index": sample.index,
        "min_ask_steps": min_ask_steps, "max_ask_steps": max_ask_steps, "ask_count": len(asks),
        "terminal_status": terminal_status, "actor_enable_thinking": thinking_enabled,
        "base_token_count": len(base_ids), "full_token_count": len(trajectory_ids),
        "response_token_budget": response_budget,
        "response_token_count": len(response_tokens), "response_mask_ones": sum(response_mask),
        "repaired_unclosed_ask_count": repaired_ask_count,
        "environment_response_tokens": len(response_mask) - sum(response_mask), "alignment": alignment, "exact_alignment": True,
        "audio_protocol_version": sample.metadata.get("audio_protocol_version"),
        "audio_environment_audits": [
            action.get("audio_environment_audit")
            for action in actions
            if action.get("audio_environment_audit") is not None
        ],
    })
    return sample


async def custom_generate(args, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    return await _custom_generate(args, sample, sampling_params, evaluation, get_observation)
