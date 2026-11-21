# BIRD-FanOut (dev 2025-11-06)

Fan-out-sensitive questions of the cleaned BIRD dev release `birdsql/bird_sql_dev_20251106`, with
corrected gold SQL where the release's gold still counts repeated rows. Built the same way as
`benchmarks/bird-fanout` (the June 2024 dev), plus the fan-out fixes the release itself made.

Fan-out: a duplicate-sensitive aggregate (COUNT, SUM, AVG without DISTINCT) or ORDER BY ... LIMIT
sees the same entity several times, because a join repeats its rows or because the counted column
repeats within its own table.

## Contents

`subset.jsonl`: one line per question, with `id` (BIRD `question_id`), `db_id`, `question`,
`evidence`, `gold_sql`, `category`, `resolvable`, the judges' `verdicts` and `reasons`, and per
category the fields below.

| category | n | resolvable | meaning |
|---|---|---|---|
| `inflated` | 33 | 33 | The release's gold still counts repeated rows where the question means distinct entities. `corrected_sql` is the corrected gold. |
| `fixed_in_release` | 36 | 35 | The release fixed a fan-out error of the June 2024 gold, keeping the question's meaning. `gold_sql` is the release's SQL, `naive_sql` the June 2024 gold (the trap its annotators fell into); `old_question` is set for the 6 the release reworded. `found_by_audit` is true for 32. |
| `intended` | 42 | 37 | The question counts the repeated rows themselves; the gold is right. Controls. |
| `contested` | 12 | 5 | Judges disagree or find the question unclear, or it has flaws beyond fan-out (`adjudication.json`). |
| `benign` | 4 | 4 | Repeated rows in form, but removing them does not change the answer on this data. |

`resolvable`: both judges say the intended reading follows from the data model: the thing the
question counts has its own key, and the data shows which rows repeat it, so an analyst who checks
keys and one-to-many relations would choose it without asking. The harness benchmark
`bird-fanout-20251106` uses the resolvable `inflated`, `fixed_in_release` and `intended` questions
(105); 102 after the gold review below.

## How it was built

1. **Audit** (`python -m misqal.fanout.bird_audit`) of all 1,534 gold queries of the release: 47 with
   definite join fan-out, 9 with an ambiguous `COUNT(*)` (only some tables repeat), 36 with a
   counted column whose values repeat.
2. **Labels** (`python -m misqal.fanout.label`): two judges (Claude Opus 5.5, GPT-5.5) label each
   flagged question inflated / intended / unclear, say whether the reading is resolvable, and write
   corrected SQL. Corrections must run and pass the audit; agreement 80/91.
3. **Release changes** (`python -m misqal.fanout.bird_changes`): every question whose gold answer
   changed between the June 2024 dev and this release, judged by the same two judges for whether
   the change fixes fan-out with the question's meaning kept.
4. **Assembly** (`python -m misqal.fanout.build_subset --changes ...`) with `adjudication.json`.

## Gold review (2026-11-06)

The corrected golds fixed fan-out only, and the intended (control) golds are BIRD's own. A Claude
agent's check, on the data, of answers that judges called "acceptable" found that many golds carry other BIRD errors
(a dropped filter, a hint formula left out, a broken join). So every harness question's gold was
reviewed for errors other than fan-out (`python -m misqal.fanout.gold_review`):

1. **Stage 1**: Gemini 3.6 Flash and Claude Haiku 5.5 review all 105 golds: correct, error (with
   fixed SQL), ambiguous, or flawed. For fan-out-corrected questions an error also gets a trap: the
   fix with the original over-count put back, so the question can still be scored as trapped.
2. **Stage 2**: Claude Opus 5.5 and GPT-5.6 Terra review every gold either first judge did not call
   correct, without seeing the first verdicts.
3. **Reconciliation**: where they disagree, each sees both reviews (judges' names hidden) and gives a
   final one.

A fix is accepted only when both strong judges call the gold an error and their fixes return the
same result; each fix must run, change the gold's result and (fan-out questions) still count each
entity once. A "fix" that returns the gold's own result counts as "correct".

A fix on which both judges agree but which returns no answer (no rows, or only 0 and NULL) is held
for a check on the data (see the train README). Here one such fix was right but returns 0, so the
question cannot show fan-out and was left out.

Result: 97 golds kept, **5 fixed** (4 on fan-out questions, all with an agreed trap), 3 left out (2
disputed, 1 without a fan-out signal). The harness benchmark now has **102 questions**. Decisions, both judges' verdicts and problems, fixes and traps:
`gold_review.jsonl`. The loader applies them; tables shown with `--known-tables` stay those of the
original golds. Cost: included in the train review.
This release's golds were already cleaned by its authors; the review flagged far fewer than on train.

## Agreement with the release's own review

The audit ran on the June 2024 dev before this release was compared. The release changed the gold
answer of 44% of the questions the audit flags as definite fan-out, against 16% of unflagged
questions. Both judges call 36 release changes fan-out fixes with the meaning kept; the audit had
flagged 32 of them. Of the 64 June 2024 questions labelled inflated, the release's new gold matches
our corrected answer exactly for 23; about 30 are unchanged and remain inflated here.
