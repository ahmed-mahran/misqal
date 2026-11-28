"""Conversion of other benchmarks to SQLite (misqal.xbench)."""

import sqlite3
import statistics

from misqal.xbench import beaver, fiben, mysql_dump, text2sql

DUMP = """CREATE TABLE `city` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(40) CHARACTER SET utf8 DEFAULT NULL COMMENT 'city name',
  `pop` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `by_name` (`name`(10))
) ENGINE=InnoDB;
INSERT INTO `city` VALUES (1,'O\\'Hare',2.5),(2,NULL,-1e3),(3,'a,b (c)',7);
"""


def test_mysql_dump(tmp_path):
    (tmp_path / "d.sql").write_text(DUMP)
    report = mysql_dump.convert(tmp_path / "d.sql", tmp_path / "d.sqlite")
    assert report == {"city": (3, 3)}
    rows = sqlite3.connect(tmp_path / "d.sqlite").execute("SELECT * FROM city ORDER BY id").fetchall()
    assert rows == [(1, "O'Hare", 2.5), (2, None, -1000.0), (3, "a,b (c)", 7.0)]


def test_text2sql_substitutes_longer_names_first():
    entry = {"sql": ["SELECT * FROM f WHERE f.n = number0 AND f.fn = flight_number0"],
             "variables": [{"name": "number0", "example": "1"}, {"name": "flight_number0", "example": "99"}]}
    q, sql = text2sql.substitute(entry, {"text": "flight flight_number0", "variables": {"flight_number0": ""}})
    assert q == "flight 99" and sql.endswith("f.n = 1 AND f.fn = 99")


def test_beaver_variance_keeps_repetition():
    (sql,), tags = beaver.transpile("SELECT VARIANCE(x) AS v, STDDEV_SAMP(x) AS s FROM t")
    assert "moments" in tags
    xs = [1, 1, 1, 4, 9]  # repeated rows weigh in, as in MySQL
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (x)")
    con.executemany("INSERT INTO t VALUES (?)", [(x,) for x in xs])
    v, s = con.execute(sql).fetchone()
    assert abs(v - statistics.pvariance(xs)) < 1e-9 and abs(s - statistics.stdev(xs)) < 1e-9


def test_fiben_transpile():
    sql = fiben.transpile("SELECT YEAR(o.D) FROM FIBEN.ORDERS o")
    assert "STRFTIME('%Y'" in sql and "FIBEN." not in sql
