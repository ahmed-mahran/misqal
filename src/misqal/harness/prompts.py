"""Prompts for the experiment arms.

Arms are ``evidence`` × ``procedure`` (× delivery: in the prompt, or ``-tool``). The paper's arms:

| paper           | code                  | what the model gets                                          |
|-----------------|-----------------------|--------------------------------------------------------------|
| no evidence     | ``none-minimal``      | the schema only                                              |
| Stats           | ``stats-minimal``     | rows, distinct values and NULLs per non-measure column       |
| Stats+Rep       | ``stats2-minimal``    | Stats plus rows per identifier value                         |
| FD              | ``full-minimal``      | the FD profile (``CardinalityProfile.to_markdown()``)        |
| FD (tool)       | ``facts-minimal-tool``| the same facts as sentences, from ``cardinality_matrix``     |
| Grain-FD        | ``full2-minimal``     | FD plus repetition counts, NULL-aware FDs, measure grain     |
| Grain-FD (tool) | ``facts2-minimal-tool``| the same additions as sentences, keys always included       |

Not used in the paper (explored early): evidence ``labels`` (pair labels only) and procedure
``grain`` (a step-by-step grain procedure the agent fills in before its final SQL).
"""

from __future__ import annotations

from misqal.cardinality.models import CardinalityProfile

BASE = """You answer questions about a SQLite database using the `execute_sql` tool.
Every number you report must come from a query you ran in this turn. If a query fails, check the
column and table names against the schema below and fix them. Do not assume how tables and columns
relate from their names or from what seems typical: check it in the data, and treat a relationship
you have not checked as unknown. Finish by calling `submit_answer` with the exact SQL whose result
answers the question and a short answer in words. The submitted SQL must select exactly what the
question asks for -- no extra columns (e.g. no counts or ids unless asked) -- with one value per
cell (no lists or concatenated values in a column).
{procedure}
## Database schema
{schema}"""

MATRIX_TOOL_DOC = """Analyze the cardinality relationships between columns of ONE base table (no joins).

Computes, from the data, whether each pair of the given columns is 1:1, N:1, 1:N or M:N, which
columns are row keys or constants, whether the column combination is unique, and hierarchies.
Numbers are compared programmatically (no LLM arithmetic).

Pass the table name and 2-25 columns from that table that your query relies on; include the
table's row-identifying column when it exists. Run separately for each table."""

LABELS_TOOL_DOC = """Report the cardinality between pairs of columns of ONE base table (1:1, N:1, 1:N or
M:N), computed from the data. Pass the table name and 2-25 columns from that table."""

FACTS_TOOL_DOC = """Describe how columns of ONE base table relate, computed from the data: for each
column, which columns have exactly one value per value of it and which can have many. Pass the table
name and 2-25 columns from that table."""

GRAIN_PROCEDURE = """
## Cardinality & grain protocol (MANDATORY)

**Theory (relational algebra):** every query has a *calculation grain* (G_calc), where measures are
computed, and an *output grain* (G_out), where results are projected. If G_out contains dimensions
not in G_calc and those dimensions are **1:N or M:N** from G_calc, a level-of-detail (LOD) mismatch
exists: joining or grouping by them multiplies rows and inflates SUM/COUNT/AVG, and changes which
rows a rank / top-N keeps.
{evidence_rules}
**Calculus of computed grain (intermediate results):** a CTE or subquery is a new relation. Its
uniqueness can only be proven by: (1) **empirical proof** – the cardinality evidence shows the
columns are unique for that table; (2) **constructive proof** – it uses GROUP BY, so it is unique on
the GROUP BY columns; (3) **inheritance proof** – it only filters (WHERE) without dropping columns.
**Projection destroys uniqueness:** selecting a subset of columns without aggregating does NOT keep
uniqueness; unique on (A,B,C) never implies unique on (B,C). **Safe alignment:** before joining two
relations, each must be proven unique on the join key; otherwise aggregate it to the join key first.

### Three-Bucket algorithm (write this plan before the final SQL)
Symbols: G_calc = D_partition ∪ D_subject; G_out = final row grain.
1. **SUBJECT:** "top N **X**", "rank **X**", "total per **X**" → X's identifier in D_subject.
2. **PARTITION:** "per **Y**", "in each **Y**", "within **Y**" (when Y bounds a ranking) → D_partition.
3. **PASS-THROUGH:** every other output dimension → D_passthrough.
D_partition and D_subject come from the question only, never from the list of output columns.

| Pattern | Request | partition | subject | passthrough | fan_out |
|---|---|---|---|---|---|
| A Ranked child | "Top N <Child> in each <Parent> by <M>, showing <Child_Attr>" | [Parent] | [Child] | [Child_Attr] | true |
| B Aggregation + orthogonal | "Total <M> per <Entity>, include <Ortho_ID>" | [] | [Entity] | [Ortho_ID] | true |
| C Global ranking | "Top N <Entity> by <M>, showing <Property>" | [] | [Entity] | [Property] | true |
| D Simple aggregation | "Total <M> by <Dim_A> and <Dim_B>" | [] | [Dim_A, Dim_B] | [] | false |
| E Global scalar | "Overall total <M>" | [] | [] | [] | false |
| F Nested hierarchy | "Top N <GrandChild> in each <Parent>, show <Child>, <Attr>" | [Parent] | [GrandChild] | [Child, Attr] | true |
| G Window share | "<Entity>'s share of total within <Partition>" | [Partition] | [Entity] | [] | false |
| H Time comparison | "Compare <M> for <Entity> across <Time>, show <Attr>" | [Time] | [Entity] | [Attr] | true |

**Cardinality check:** for EACH D_passthrough column, check G_calc → column. It is safe only when
N:1 or 1:1 is explicitly proven; 1:N or M:N means fan-out. **Missing evidence ⇒ assume fan-out**;
never infer N:1 or 1:1 from column names. **requires_fan_out_join** = TRUE if any pass-through is
proven 1:N/M:N from G_calc or lacks proof of N:1/1:1.

**Strategy:**
- requires_fan_out_join = false → standard aggregation at G_calc is safe.
- true, and one row per G_calc entity is wanted → **Dimensional folding:** aggregate at G_calc and
  fold pass-through columns into a set/list (e.g. GROUP_CONCAT(DISTINCT ...)); apply rank/limit after.
- true, and G_out is finer than G_calc → **Safe expansion:** compute the measure and apply the
  rank/filter at G_calc first, then join the surviving rows to a deduplicated attribute relation
  (join key + pass-through columns) to attach D_passthrough.
- Never put D_passthrough in the calculation GROUP BY or a window PARTITION BY. Never join a measure
  CTE to a base table carrying dimensions outside G_calc ∪ D_passthrough.

### Five laws of grain
I. **Deterministic grain:** every relation has a definite grain (unique key); establish it before
joining; deduplicate/compress unstable inputs first.
II. **Pre-aggregation:** joining orders to line items before SUM(order revenue) multiplies each order
total by its line count. Aggregate each measure side to the shared grain separately, then join the
compressed 1:1 results. Do not pull many-side columns through a join when aggregating a one-side total.
III. **Vertical conservation:** never re-aggregate one-side attributes after joining to the many side;
a parent total copied to N child rows and then summed is multiplied by N.
IV. **Domain completeness:** when aligning two or more measure subsets, use a full outer join on the
shared keys (after pre-aggregation); inner/left joins silently drop keys present on one side only.
V. **Display vs consumption:** computing at a coarse grain while displaying a finer one needs
decoupling (window partitioned by the coarse grain only, or pre-aggregate and join back).

**Non-commutative expansion:** rank, row_number, limit, percentile and top-N do not commute with
1:N/M:N joins: apply them at G_calc BEFORE any fan-out join.
**Expected fan-out:** when a top-N result legitimately projects 1:N attributes, the row count exceeds
N. That is correct; do not "fix" it by grouping by the pass-through attributes.
"""

GRAIN_EVIDENCE_RULES = """
**When to run `cardinality_matrix`:** when the query (a) uses at least two distinct dimension columns
from a table, or (b) joins two or more tables. Run it BEFORE writing the final SQL, on EVERY table in
the query, in isolation, passing every dimension column that participates (grain, ranking/partition,
output, join keys) plus the table's row-identifying column. Do not run it when dimensions are used
only for filtering. Never assert unique grain or a safe join without that evidence.
"""

GRAIN_INJECTED_EVIDENCE_RULES = """
**Cardinality evidence** for every table of the database is given with the question, one block per
table (computed from the data). Use it for every join and every pass-through column; never assert unique
grain or a safe join that the evidence does not show.
"""

GRAIN_NO_EVIDENCE_RULES = """
No cardinality tool is available here: relationships between columns are unproven unless the schema
declares them (primary keys). Apply the epistemic default below — unproven ⇒ assume fan-out.
"""


TOOL_MANDATE = """
## Cardinality evidence (MANDATORY)
Before writing your final SQL you MUST run `cardinality_matrix` on EVERY table your query reads,
passing the columns your query relies on. Base every relationship between columns that your query
depends on, on its result. A submission that reads a table you have not run `cardinality_matrix`
on is rejected.
"""


def system_prompt(schema: str, evidence: str, procedure: str, delivery: str = "inject") -> str:
    proc = TOOL_MANDATE if evidence != "none" and delivery == "tool" else ""
    if procedure == "grain":
        if evidence == "none":
            rules = GRAIN_NO_EVIDENCE_RULES
        else:
            rules = GRAIN_EVIDENCE_RULES if delivery == "tool" else GRAIN_INJECTED_EVIDENCE_RULES
        proc += GRAIN_PROCEDURE.format(evidence_rules=rules)
    elif procedure != "minimal":
        raise ValueError(f"unknown procedure {procedure!r}")
    return BASE.format(procedure=proc, schema=schema)


def render_profile(profile: CardinalityProfile, evidence: str) -> str:
    if evidence == "full":
        return profile.to_markdown()
    if evidence == "labels":
        return profile.to_markdown(relations_only=True)
    if evidence == "facts":
        return profile.to_sentences()
    raise ValueError(f"unknown evidence render {evidence!r}")
