# BIRD-FanOut (train)

Fan-out-sensitive questions of the BIRD train split (9,428 questions, 69 SQLite databases), with
corrected gold SQL where the gold counts repeated rows. Built with the same audit and labelling
pipeline as `benchmarks/bird-fanout-20251106` (the cleaned dev release), plus an escalation step that
sends the cases the first judges left unsure to stronger judges.

Fan-out: a duplicate-sensitive aggregate (COUNT, SUM, AVG without DISTINCT) or ORDER BY ... LIMIT
sees the same entity several times, because a join repeats its rows or because the counted column
repeats within its own table.

## Contents

`subset.jsonl`: one line per audited question, with `id` (position in BIRD's `train.json`, which
has no question ids), `db_id`, `question`, `evidence`, `gold_sql`, `category`, `resolvable`,
`escalated`, the judges' `verdicts` and `reasons`, and for `inflated` rows `corrected_sql`.

| category | resolvable | not resolvable | meaning |
|---|---|---|---|
| `inflated` | 255 | 4 | The gold counts repeated rows where the question means distinct entities. `corrected_sql` is the corrected gold: both judges' corrections return the same result. |
| `intended` | 166 | 39 | The question counts the repeated rows themselves; the gold is right. Controls. |
| `benign` | 45 | 0 | Repeated rows in form, but removing them does not change the answer on this data. |
| `contested` | 15 | 35 | The judges disagree or find the question unclear. |
| `unresolved` | 25 | 2 | Both judges say inflated, but their corrections return different results (or one does not run). |

`resolvable`: both judges say the intended reading follows from the data model (the counted thing
has its own key, and the data shows which rows repeat it). The harness benchmark
`bird-fanout-train` uses the resolvable `inflated` and `intended` questions: **421 questions over 56
databases** (255 inflated, 166 intended). After the gold review below, 384 (229 inflated, 155 intended).

Of the 205 inflated questions with a single-number answer, the gold is a median 1.6× the corrected
answer; 53 are below it (e.g. percentages with child-row denominators) and the largest ratio is
about 27,000×.

## How it was built

1. **Audit** (`python -m misqal.fanout.bird_audit --root <BIRD train>`), all 9,428 gold queries, a
   few minutes: 586 with definite (realized) join fan-out (6.2%), 177 more with only an ambiguous
   `COUNT(*)`, 711 more with only a counted column whose values repeat. 187 gold queries do not
   parse against the schema; 178 of them are in `retail_world`, where 330 of 373 gold queries do not
   run in SQLite. Only the 586 definite flags were labelled.
2. **Judge choice.** Candidate judges were scored on the 93 labelled questions of the June 2024 dev
   audit (`benchmarks/bird-fanout`). Gemini 3.6 Flash (98% verdicts, 97% of corrections equal the
   reference answer) and Claude Haiku 5.5 (95%, 94%) were chosen as first judges.
3. **Labels** (`python -m misqal.fanout.label`): both first judges label each flagged question
   inflated / intended / unclear, say whether it is resolvable, and write corrected SQL. Agreement
   515/586.
4. **Escalation.** The 141 questions the first judges left unsure (they disagree or are unclear;
   both say inflated but not both resolvable; or their corrections return different results) were
   labelled again by Claude Opus 5.5 and GPT-5.6 Terra, whose labels replace the first judges' for
   those questions (`escalated: true`).
5. **Assembly** (`python -m misqal.fanout.build_subset --escalated ...`) with an empty
   `adjudication.json`: no hand adjudication.

Labelling cost about $15 in total (judge choice $5.90, first judges $4.50, escalation $4.75).

## Gold review (2026-11-06)

The corrected golds fixed fan-out only, and the intended (control) golds are BIRD's own. A Claude
agent's check, on the data, of answers that judges called "acceptable" found that many golds carry other BIRD errors
(a dropped filter, a hint formula left out, a broken join). So every harness question's gold was
reviewed for errors other than fan-out (`python -m misqal.fanout.gold_review`):

1. **Stage 1**: Gemini 3.6 Flash and Claude Haiku 5.5 review all 421 golds: correct, error (with
   fixed SQL), ambiguous, or flawed. For fan-out-corrected questions an error also gets a trap: the
   fix with the original over-count put back, so the question can still be scored as trapped.
   The review flagged all 21 bad golds that the agent's check had found.
2. **Stage 2**: Claude Opus 5.5 and GPT-5.6 Terra review every gold either first judge did not call
   correct, without seeing the first verdicts.
3. **Reconciliation**: where they disagree, each sees both reviews (judges' names hidden) and gives a
   final one.

A fix is accepted only when both strong judges call the gold an error and their fixes return the
same result; each fix must run, change the gold's result and (fan-out questions) still count each
entity once. A "fix" that returns the gold's own result counts as "correct".

A fix on which both judges agree but which returns no answer (no rows, or only 0 and NULL) is not
accepted on agreement alone: it is held as `check` and verified on the data. A data check of the 12
such fixes found 3 wrong (a literal with a spacing slip both judges made; missing values stored as the
text 'NULL'; "New York" read as state or city): two were corrected on the data and one left out. Of
the 9 right ones, 7 here (and 1 in the dev split) were left out because a question whose answer is 0
or NULL cannot show fan-out; one was kept because its over-count still gives a different number.

Result: 303 golds kept, **80 fixed** (56 on fan-out questions, 39 of them with a trap; the others
have no trapped score), 37 left out (27 disputed, 10 flawed or without a fan-out signal); 1 ambiguous
kept as is. The harness benchmark now has **384 questions** (229 inflated, 155 intended). Decisions, both judges' verdicts and problems, fixes and traps:
`gold_review.jsonl`. The loader applies them; tables shown with `--known-tables` stay those of the
original golds. Cost about $15 for both splits.

## Use

```
uv run python -m misqal.harness.run --bench bird-fanout-train --root <BIRD train> \
    --model <model> --arm none-minimal facts-minimal-tool --known-tables --out runs/<name>
```

`--root` defaults to `data/bird-train/train` (BIRD's `train.json` and `train_databases/`).
`--known-tables` shows the agent only the tables read by the corrected or the original gold, with all
their columns, in every arm.

## License

Derived from BIRD (bird-bench.github.io), CC BY-SA 4.0; this subset is released under the same
license. Questions, hints and gold SQL are BIRD's; corrected SQL and labels are ours.
