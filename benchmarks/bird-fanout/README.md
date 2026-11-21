# BIRD-FanOut pilot (June 2024 dev)

The questions of the June 2024 BIRD dev release whose gold SQL is exposed to fan-out on the actual
data, with intent labels and corrected gold SQL where the gold answer is inflated. Used for pilots
and to choose the inexpensive judges for train. The paper's BIRD-FanOut is
`benchmarks/bird-fanout-20251106` (dev) plus `benchmarks/bird-fanout-train`.

Fan-out: joining a table to another on a key that is not unique there repeats its rows; COUNT /
SUM / AVG without DISTINCT then count the same entity several times, and ORDER BY … LIMIT N can
return the same entity in several slots.

## Contents

`subset.jsonl`: one line per question (93), with `id` (BIRD `question_id`), `db_id`, `question`,
`evidence`, `gold_sql`, `category`, both judges' `verdicts` and `reasons`, and for inflated
questions `corrected_sql`, `corrected_by`, `other_error` and, for single-number answers,
`gold_to_corrected` (gold value / corrected value).

| category | n | meaning |
|---|---|---|
| `inflated` | 64 | The question means distinct entities; the gold answer counts repeated rows. `corrected_sql` is the corrected gold. |
| `intended` | 21 | The question counts the joined rows themselves (e.g. "number of votes"); the gold is right. |
| `contested` | 5 | Judges disagree or find it unclear, typically because the question asks about distinct entities while the hint prescribes a row-level formula. |
| `benign` | 3 | Repeated rows in form, but removing them does not change the answer on this data. |

For 53 inflated questions with a single-number answer, gold / corrected has median 6.3 and maximum
2,718 (counting lap-time rows instead of drivers); 12 are below 1 (percentages whose denominator was
inflated). 3 corrections also fix a mistake unrelated to fan-out (`other_error`).

## How it was built

1. **Audit** (`python -m misqal.fanout.bird_audit`): all 1,534 BIRD dev gold queries; 93 have a
   duplicate-sensitive aggregate or top-N over rows that actually repeat on the data under the
   query's own filters.
2. **Labels** (`python -m misqal.fanout.label`): two judges (Claude Opus 5.5, GPT-5.5) independently
   decide intent and write corrected SQL; corrections must run and pass the audit. Agreement 88/93.
   On 20 questions also labelled by a Claude agent, each judge agreed with the agent's label on 18.
3. **Adjudication** (`adjudication.json`): where both judges call a question inflated but their
   corrections return different answers (4 questions), GPT-5.5's correction is used (it also fixes
   unrelated errors in the gold); notes explain each decision and each contested case.
4. **Assembly** (`python -m misqal.fanout.build_subset`).

## Licence

Derived from the BIRD dev set (https://bird-bench.github.io), licensed CC BY-SA 4.0; this subset is
released under the same licence.
