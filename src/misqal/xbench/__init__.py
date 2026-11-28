"""Fan-out audit of the gold SQL of other text-to-SQL benchmarks (paper §4, Figure 6, extended
version Appendix D): Spider 1.0, KaggleDBQA, Spider 2.0-Lite (SQLite part), ATIS, Advising, FIBEN
and BEAVER.

Benchmarks that ship as MySQL or Db2 dumps are converted to SQLite first:

- ``misqal.xbench.mysql_dump``: a mysqldump file to a SQLite database (ATIS, Advising, BEAVER);
- ``misqal.xbench.text2sql``: ATIS and Advising gold queries;
- ``misqal.xbench.fiben``: the FIBEN database (Db2 DDL + CSV) and gold queries;
- ``misqal.xbench.beaver``: BEAVER gold queries.

Each converter keeps only golds that run on SQLite and writes them as JSONL
(``bench, id, db_id, db, question, sql``). Then ``misqal.xbench.audit`` audits every gold, and
``misqal.xbench.table`` combines the audits with the two agent labels shipped in
``benchmarks/other-benchmarks/labels.jsonl``.
"""
