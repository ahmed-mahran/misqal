"""Deterministic synthetic databases for the fan-out trap suite.

- ``bookings``: one wide, denormalized sales table. Each product has one sub-family, and each
  sub-family one family, but some products are booked under two divisions and every product is
  booked by several corporate customers, so division / product / customer are many-to-many.
- ``supply``: products, warehouses, customers, orders, order lines (shipped from a warehouse) and
  inventory kept in lots: several rows per product and warehouse. Orders and inventory are both
  "many" per product, the setting of a chasm trap. Products and suppliers are many-to-many.
- ``shop``: one wide order-line table where order-level fields repeat per line (fan trap), with
  gaps in an attribute (NULL splits a product into two groups), many-to-many product tags, and
  sales targets at a finer grain than region.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

DIVISIONS = ["Networking", "Storage", "Wireless", "Security", "Compute", "Software"]
CUSTOMERS = ["Acme Retail", "Bluefin Logistics", "Cobalt Health", "Delta Foods", "Evergreen Bank",
             "Falcon Air", "Granite Energy", "Harbor Telecom", "Ironwood Mining", "Juniper Media",
             "Kestrel Motors", "Lumen Pharma", "Meridian Insurance", "Northwind Motors",
             "Orchid Hotels", "Pinnacle Steel"]


def build_bookings(path: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    products = []
    pid = 0
    for division in DIVISIONS:
        for f in range(2):
            family = f"{division[:3].upper()}-F{f + 1}"
            for s in range(2):
                subfamily = f"{family}-S{s + 1}"
                for _ in range(rng.randint(3, 5)):
                    pid += 1
                    products.append({"product": f"P{pid:04d}", "subfamily": subfamily,
                                     "family": family, "division": division})
    # ~15% of products are also booked under a second division
    for p in products:
        others = [d for d in DIVISIONS if d != p["division"]]
        p["divisions"] = [p["division"]] + ([rng.choice(others)] if rng.random() < 0.15 else [])
        p["corps"] = rng.sample(range(1001, 1061), rng.randint(2, 6))
        # products sold across two divisions tend to be big flagship products
        p["scale"] = rng.lognormvariate(0, 0.9) * (3.0 if len(p["divisions"]) > 1 else 1.0)
    weeks = [f"FY26 W{w:02d}" for w in range(1, 9)]
    orgs = ["Americas", "EMEA", "APJ"]
    rows = []
    for i in range(6000):
        p = rng.choice(products)
        rows.append((i + 1, rng.choice(p["divisions"]), p["family"], p["subfamily"], p["product"],
                     rng.choice(p["corps"]), rng.choice(orgs), rng.choice(weeks),
                     round(rng.uniform(500, 20000) * p["scale"], 2)))
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as con:
        con.execute("""CREATE TABLE bookings (
            booking_id INTEGER PRIMARY KEY, division TEXT, product_family TEXT,
            product_subfamily TEXT, product TEXT, corp_id INTEGER, sales_org TEXT,
            booking_week TEXT, amount_usd REAL)""")
        con.executemany("INSERT INTO bookings VALUES (?,?,?,?,?,?,?,?,?)", rows)


def build_supply(path: Path, seed: int = 11) -> None:
    rng = random.Random(seed)
    categories = ["Safety", "Chassis", "Interior", "Electrical", "Powertrain"]
    products = [(i + 1, f"Part {i + 1:03d}", categories[i % len(categories)],
                 round(rng.uniform(5, 400), 2)) for i in range(40)]
    warehouses = [(1, "North"), (2, "South"), (3, "East"), (4, "West")]
    customers = [(i + 1, name) for i, name in enumerate(CUSTOMERS)]
    inventory, lot = [], 0
    for pid, *_ in products:
        for wid, _ in rng.sample(warehouses, rng.randint(1, 3)):
            for _ in range(rng.randint(1, 4)):  # several lots per product and warehouse
                lot += 1
                inventory.append((lot, pid, wid, rng.randint(0, 300)))
    stocked = {(p, w) for _, p, w, _ in inventory}
    orders, lines, line_id = [], [], 0
    for oid in range(1, 801):
        cust = rng.choice(customers)[0]
        orders.append((oid, cust, f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}"))
        for _ in range(rng.randint(1, 5)):
            pid, wid = rng.choice(sorted(stocked))
            line_id += 1
            lines.append((line_id, oid, pid, wid, rng.randint(1, 20)))
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT,
                                   category TEXT, unit_price REAL);
            CREATE TABLE warehouses (warehouse_id INTEGER PRIMARY KEY, warehouse_name TEXT);
            CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, customer_name TEXT);
            CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER,
                                 order_date TEXT);
            CREATE TABLE order_lines (line_id INTEGER PRIMARY KEY, order_id INTEGER,
                                      product_id INTEGER, warehouse_id INTEGER, quantity INTEGER);
            CREATE TABLE inventory (lot_id INTEGER PRIMARY KEY, product_id INTEGER,
                                    warehouse_id INTEGER, on_hand INTEGER);
        """)
        con.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
        con.executemany("INSERT INTO warehouses VALUES (?,?)", warehouses)
        con.executemany("INSERT INTO customers VALUES (?,?)", customers)
        con.executemany("INSERT INTO orders VALUES (?,?,?)", orders)
        con.executemany("INSERT INTO order_lines VALUES (?,?,?,?,?)", lines)
        con.executemany("INSERT INTO inventory VALUES (?,?,?,?)", inventory)
    # suppliers come from their own generator so the tables above do not depend on them
    srng = random.Random(seed + 1000)
    suppliers = [(i + 1, f"Supplier {chr(65 + i)}") for i in range(8)]
    product_suppliers = [(pid, sid) for pid, *_ in products
                         for sid in sorted(srng.sample(range(1, 9), srng.randint(1, 3)))]
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE suppliers (supplier_id INTEGER PRIMARY KEY, supplier_name TEXT);
            CREATE TABLE product_suppliers (product_id INTEGER, supplier_id INTEGER,
                                            PRIMARY KEY (product_id, supplier_id));
        """)
        con.executemany("INSERT INTO suppliers VALUES (?,?)", suppliers)
        con.executemany("INSERT INTO product_suppliers VALUES (?,?)", product_suppliers)


def build_shop(path: Path, seed: int = 13) -> None:
    """One wide order-line table: order-level fields (date, customer, region, shipping fee) repeat
    on every line, some customers ship to two regions, and some products miss their category on
    part of their rows. Product tags are many-to-many; sales targets are per region, segment and
    quarter."""
    rng = random.Random(seed)
    categories = ["Home", "Garden", "Kitchen", "Outdoor", "Toys", "Office"]
    tags = ["eco", "premium", "bestseller", "clearance", "new"]
    products = []
    for i in range(30):
        # a third of the products have their category missing on a quarter of their rows;
        # these tend to be the high-volume imports
        gappy = i % 3 == 0
        products.append({"name": f"Item {i + 1:02d}", "category": categories[i % len(categories)],
                         "price": round(rng.uniform(4, 120), 2), "gappy": gappy,
                         "weight": rng.lognormvariate(0, 0.7) * (2.0 if gappy else 1.0),
                         "tags": rng.sample(tags, rng.randint(1, 3))})
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "SMB", "Enterprise"]
    customers = [{"name": name, "segment": segments[i % 3],
                  "regions": rng.sample(regions, 2 if rng.random() < 0.4 else 1)}
                 for i, name in enumerate(CUSTOMERS + ["Quarry Labs", "Redwood Clinics",
                                                       "Summit Outfitters", "Tidewater Co"])]
    weights = [p["weight"] for p in products]
    lines, line_id = [], 0
    for oid in range(1, 1201):
        c = rng.choice(customers)
        date = f"2026-{rng.randint(1, 6):02d}-{rng.randint(1, 28):02d}"
        region = rng.choice(c["regions"])
        fee = round(rng.choice([0, 4.99, 9.99, 14.99, 24.99]), 2)
        for p in rng.choices(products, weights, k=rng.randint(1, 6)):
            line_id += 1
            category = None if p["gappy"] and rng.random() < 0.25 else p["category"]
            lines.append((line_id, oid, date, c["name"], c["segment"], region, p["name"],
                          category, rng.randint(1, 12), p["price"], fee))
    product_tags = [(p["name"], t) for p in products for t in sorted(p["tags"])]
    targets = [(r, s, q, round(rng.uniform(20000, 60000), 2))
               for r in regions for s in segments for q in ["2026-Q1", "2026-Q2"]]
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE sales_lines (line_id INTEGER PRIMARY KEY, order_id INTEGER,
                order_date TEXT, customer_name TEXT, customer_segment TEXT, ship_region TEXT,
                product_name TEXT, category TEXT, quantity INTEGER, unit_price REAL,
                shipping_fee REAL);
            CREATE TABLE product_tags (product_name TEXT, tag TEXT,
                                       PRIMARY KEY (product_name, tag));
            CREATE TABLE sales_targets (region TEXT, customer_segment TEXT, quarter TEXT,
                                        target_revenue REAL,
                                        PRIMARY KEY (region, customer_segment, quarter));
        """)
        con.executemany("INSERT INTO sales_lines VALUES (?,?,?,?,?,?,?,?,?,?,?)", lines)
        con.executemany("INSERT INTO product_tags VALUES (?,?)", product_tags)
        con.executemany("INSERT INTO sales_targets VALUES (?,?,?,?)", targets)


BUILDERS = {"bookings": build_bookings, "supply": build_supply, "shop": build_shop}


def ensure_databases(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, build in BUILDERS.items():
        path = root / f"{name}.sqlite"
        if not path.exists():
            build(path)
        out[name] = path
    return out
