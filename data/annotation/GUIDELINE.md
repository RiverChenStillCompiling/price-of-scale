# Annotation guideline

Each item shows what a user just said to their assistant (`cue`) and an old note from past conversations that the
assistant's memory module brought up on its own (`memory`). Label the note with one digit.

| label | meaning |
|---|---|
| 1 | **Helpful**: when responding to what the user said, the assistant would use concrete content from the note, including a partial answer (one amount of several that add up to a total, or a date needed for a time span). Possibly related background does not count. |
| 2 | **Same topic, not useful**: the topic matches, but the response would not use the note; bringing it up is an interruption. |
| 3 | **Unrelated**: little to do with the cue, for example only a shared word or name. |
| 4 | **Cannot tell**: only if the item cannot be decided. |

Every LongMemEval session carries a date, so a note saying that something happened "today" supplies a date.

## Files

- `items.jsonl`: 270 items. 240 are false alarms of the recall-v1 gate on LongMemEval-M (80 each at store sizes
  64, 512 and 2048) and 30 are evidence turns mixed in as checks (`kind`). `cue_zh` and `memory_zh` are the Chinese
  machine translations shown to the human annotator (Qwen2.5-14B, cue and memory translated together, proper nouns
  kept in English).
- `label_human`: one author, a non-native English speaker who read the English text with the translation alongside.
- `label_claude`: Claude Opus 5.5. `label_gpt`: GPT-6 Sol, given the English text and this guideline in a fresh session.
- `in_agreement_set`: the 229 items used for the agreement statistics. Items 0-37 are excluded because an earlier
  translation pass contained errors and the Claude annotator saw the human labels for them; 5 further items were
  marked "cannot tell" by the human annotator.

Text in `cue` and `memory` comes from LongMemEval (Wu et al., ICLR 2025), released under the MIT licence.
