# Fan-out in other benchmarks' gold

Labels behind the paper's cross-benchmark result (§4, Figure 6; extended version, Appendix D): the
fan-out audit run on the gold SQL of seven more benchmarks, and two labels for each gold it flags as
definite fan-out.

`labels.jsonl`: one line per labelled flag (484), with `bench`, `id`, `db_id`, `question`, `gold_sql`
(SQLite; transpiled for ATIS, Advising, FIBEN and BEAVER), `first_label` and `second_label`, and
`distorted`. Each label has `label` (`distorted`, `intended` or `unclear`), `answer_changes`, `reason`
and, where given, `corrected_sql` and `resolvable`.

- **distorted**: the question means each entity once, and a corrected query (COUNT(DISTINCT …),
  aggregating before the join, de-duplicating before LIMIT, …) returns a different answer on the data.
  Duplicate rows and keys shared by several dimension rows count as fan-out; a change in row order
  alone does not.
- **intended**: the question means the joined rows, or the repetition does not change the answer on
  this data (rows repeat only across groups, uniformly under AVG, or under an existence test).
- **unclear**: both readings are plausible and give different answers.

A gold is `distorted` when both labels say so. Two Claude agents labelled each flag independently,
running the gold and their correction on the data; they agree on 455 of the 484 labels (94%). No
person labelled or checked them. Every definite flag is labelled, except BEAVER's generated warehouse
questions: a random 200 of their 1,186 definite flags.

| benchmark | golds | join | definite | labelled | agree | distorted | % golds | % joined |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Spider 1.0 dev | 1,034 | 39% | 1 | 1/1 | 1/1 | 0 | 0.00 | 0.00 |
| Spider 1.0 test | 2,147 | 40% | 20 | 20/20 | 20/20 | 4 | 0.19 | 0.46 |
| Spider 1.0 train | 6,900 | 39% | 47 | 47/47 | 41/47 | 12 | 0.17 | 0.45 |
| KaggleDBQA | 272 | 16% | 3 | 3/3 | 3/3 | 1 | 0.37 | 2.27 |
| Spider 2.0-Lite (SQLite) | 24 | 92% | 2 | 2/2 | 2/2 | 1 | 4.17 | 4.55 |
| ATIS | 4,851 | 93% | 0 | 0/0 | 0/0 | 0 | 0.00 | 0.00 |
| Advising | 4,334 | 85% | 26 | 26/26 | 14/26 | 0 | 0.00 | 0.00 |
| FIBEN | 227 | 100% | 1 | 1/1 | 1/1 | 0 | 0.00 | 0.00 |
| BEAVER warehouse, query logs | 92 | 100% | 26 | 26/26 | 24/26 | 22 | 23.91 | 23.91 |
| BEAVER warehouse, generated | 5,266 | 100% | 1,186 | 200/1186 | 192/200 | ~480 (403–562) | ~9.1 | ~9.1 |
| BEAVER network, generated | 842 | 100% | 158 | 158/158 | 157/158 | 87 | 10.33 | 10.33 |

For comparison, 4.2% of BIRD dev's golds are distorted (5.6% of those that join). Spider 1.0 train
leaves out `music_2`, whose database ships without rows, and Spider's `train_others` is not audited
for the same reason. Spider 2.0-Lite counts only its SQLite questions with public gold. EHRSQL could not
be tested: every join in it that could repeat rows goes to a subquery, which the audit checks only
statically.

## Reproducing

The data layout and the conversion of ATIS, Advising, FIBEN and BEAVER to SQLite are described in
`misqal.xbench` (`src/misqal/xbench/`). Then:

```bash
uv run python -m misqal.xbench.audit spider1-dev spider1-test spider1-train kaggledbqa spider2-lite \
    atis advising fiben beaver-dw beaver-dw_real beaver-neutron
uv run python -m misqal.xbench.table
```

## Sources and license

Spider 1.0 (Yu et al., 2018) and KaggleDBQA (Lee et al., 2021): CC BY-SA 4.0. Advising
(Finegan-Dollak et al., 2018): CC BY 4.0, from the `jkkummerfeld/text2sql-data` release; ATIS, from the
same release, had no definite flag, so nothing of it is included here. FIBEN (Sen et al., 2020):
Apache-2.0. BEAVER (Chen et al., 2024): MIT. Spider 2.0-Lite (Lei et al., 2025): MIT. Questions and gold
SQL are the benchmarks'; labels, corrections and reasons are ours. Because it includes material from
CC BY-SA 4.0 benchmarks, this file is released under CC BY-SA 4.0.
