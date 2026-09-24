# Teacher trajectory bank schema

Each JSONL row contains one answer-verified training trajectory. Required top-level fields are `trajectory_id`, `sample_id`, `question`, `choices`, `initial_description`, `correct_letter`, `ask_count`, `ask_bucket`, `actions`, `same_actor_prompt_as_rl`, `prompt_alignment`, and `training_target_policy`.

Each `ask` action contains `canonical_action`, `query`, and `observation`. The final action is an `answer` whose label equals `correct_letter`. `training_target_policy` is either `canonical_actions_only` or `raw_teacher_actions`; the latter is accepted only when the row declares exact actor-prompt alignment.
