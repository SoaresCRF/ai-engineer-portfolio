"""Synthetic orders in SQLite: schema, deterministic generator and lookup.

Money is in integer cents; timestamps are ISO 8601 UTC with a "Z" suffix.
"""

import argparse
import random
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from support_agent.policy import (
    COMPLAINT_WINDOW_DAYS,
    DELIVERY_MODES,
    LATE_COUPON_MINUTES,
    STORE_TYPES,
)

REFERENCE_NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
STATUSES = ("em_preparo", "a_caminho", "entregue", "nao_entregue", "cancelado")
HISTORY_DAYS = 45
COUPON_PROBABILITY = 0.2
LATE_PROBABILITY = 0.35
MAX_LATE_MINUTES = 90
MAX_EARLY_MINUTES = 20
RECENT_MAX_MINUTES = 180  # em_preparo / a_caminho are created within the last 3 hours
ORDER_ID_RANGE = (10000, 100000)  # extra_orders must use ids outside it (e.g. P0xxxx)

# Same order as STATUSES; the edge orders below force "entregue".
STATUS_WEIGHTS = (6, 6, 70, 8, 10)
DELIVERY_FEES = (0, 499, 799, 990)
SERVICE_FEES = (0, 150, 250)
COUPONS = (500, 1000, 1500)
NOTES = ("Sem cebola", "Deixar na portaria", "Tocar a campainha", "Sem talheres")
NOTE_PROBABILITY = 0.15

FIRST_NAMES = (
    "Ana", "Bruno", "Carla", "Diego", "Elisa", "Felipe", "Gabriela", "Hugo", "Iara", "João",
)  # fmt: skip
LAST_NAMES = ("Almeida", "Barbosa", "Cardoso", "Dias", "Esteves", "Ferreira", "Gomes", "Lima")

# (item name, unit price in cents) per store type.
CATALOG = {
    "restaurante": (("Pizza de calabresa", 4200), ("Açaí 500ml", 1800), ("Hambúrguer", 3000)),
    "mercado": (("Arroz 5kg", 2800), ("Detergente", 350), ("Leite integral 1L", 600)),
    "varejo": (("Fone de ouvido", 5900), ("Camiseta", 4500), ("Garrafa térmica", 6500)),
}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS customers (id TEXT PRIMARY KEY, nome TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    store_type TEXT NOT NULL CHECK (store_type IN {STORE_TYPES}),
    delivery_mode TEXT NOT NULL CHECK (delivery_mode IN {DELIVERY_MODES}),
    status TEXT NOT NULL CHECK (status IN {STATUSES}),
    created_at TEXT NOT NULL,
    promised_at TEXT NOT NULL,
    delivered_at TEXT,
    delivery_fee_cents INTEGER NOT NULL CHECK (delivery_fee_cents >= 0),
    service_fee_cents INTEGER NOT NULL CHECK (service_fee_cents >= 0),
    discount_cents INTEGER NOT NULL DEFAULT 0 CHECK (discount_cents >= 0),
    total_cents INTEGER NOT NULL CHECK (total_cents >= 0),
    observacao TEXT
);
CREATE TABLE IF NOT EXISTS items (
    order_id TEXT NOT NULL REFERENCES orders(id),
    nome TEXT NOT NULL,
    qty INTEGER NOT NULL CHECK (qty > 0),
    unit_cents INTEGER NOT NULL CHECK (unit_cents >= 0)
);
CREATE TABLE IF NOT EXISTS refunds (
    protocolo TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE REFERENCES orders(id),
    motivo TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""

ORDER_COLUMNS = (
    "id", "customer_id", "store_type", "delivery_mode", "status", "created_at", "promised_at",
    "delivered_at", "delivery_fee_cents", "service_fee_cents", "discount_cents", "total_cents",
    "observacao",
)  # fmt: skip


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def get_order(conn: sqlite3.Connection, order_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if row is None:
        return None
    items = conn.execute(
        "SELECT nome, qty, unit_cents FROM items WHERE order_id = ? ORDER BY rowid", (order_id,)
    ).fetchall()
    refund = conn.execute("SELECT * FROM refunds WHERE order_id = ?", (order_id,)).fetchone()
    return {
        **dict(row),
        "items": [dict(i) for i in items],
        "refund": dict(refund) if refund else None,
    }


def _insert_order(conn: sqlite3.Connection, order: dict) -> None:
    values = {"discount_cents": 0, "observacao": None, "delivered_at": None, **order}
    marks = ", ".join("?" * len(ORDER_COLUMNS))
    conn.execute(
        f"INSERT INTO orders ({', '.join(ORDER_COLUMNS)}) VALUES ({marks})",
        [values[c] for c in ORDER_COLUMNS],
    )
    conn.executemany(
        "INSERT INTO items (order_id, nome, qty, unit_cents) VALUES (?, ?, ?, ?)",
        [(order["id"], i["nome"], i["qty"], i["unit_cents"]) for i in order["items"]],
    )


def _edge_slots(rng: random.Random) -> list[tuple[str, float]]:
    """(store_type, days before now of delivery): one inside and one outside each window."""
    slots = []
    for store, days in COMPLAINT_WINDOW_DAYS.items():
        slots.append((store, rng.uniform(0.1, days * 0.9)))
        slots.append((store, rng.uniform(days + 0.5, HISTORY_DAYS - 0.5)))
    return slots


def _timeline(rng: random.Random, now: datetime, status: str, delivered_days: float | None) -> dict:
    if status in ("em_preparo", "a_caminho"):
        created = now - timedelta(minutes=rng.randint(5, RECENT_MAX_MINUTES))
    elif status == "entregue":
        days = (
            delivered_days if delivered_days is not None else rng.uniform(0.1, HISTORY_DAYS - 0.5)
        )
        delivered = now - timedelta(days=days)
        late = rng.random() < LATE_PROBABILITY
        delay = (
            rng.randint(LATE_COUPON_MINUTES + 1, MAX_LATE_MINUTES)
            if late
            else rng.randint(-MAX_EARLY_MINUTES, LATE_COUPON_MINUTES - 1)
        )
        promised = delivered - timedelta(minutes=delay)
        created = promised - timedelta(minutes=rng.randint(30, 60))
        return {
            "created_at": to_iso(created),
            "promised_at": to_iso(promised),
            "delivered_at": to_iso(delivered),
        }
    else:
        created = now - timedelta(days=rng.uniform(0.1, HISTORY_DAYS - 0.5))
    promised = created + timedelta(minutes=rng.randint(30, 90))
    return {"created_at": to_iso(created), "promised_at": to_iso(promised)}


def _build_order(
    rng: random.Random, order_id: str, customer_id: str, now: datetime, edge: tuple | None
) -> dict:
    store = edge[0] if edge else rng.choice(STORE_TYPES)
    status = "entregue" if edge else rng.choices(STATUSES, STATUS_WEIGHTS)[0]
    picked = [rng.choice(CATALOG[store]) for _ in range(rng.randint(1, 4))]
    items = [{"nome": n, "qty": rng.randint(1, 3), "unit_cents": p} for n, p in picked]
    subtotal = sum(i["qty"] * i["unit_cents"] for i in items)
    discount = min(rng.choice(COUPONS), subtotal) if rng.random() < COUPON_PROBABILITY else 0
    fees = (rng.choice(DELIVERY_FEES), rng.choice(SERVICE_FEES))
    return {
        "id": order_id,
        "customer_id": customer_id,
        "store_type": store,
        "delivery_mode": rng.choice(DELIVERY_MODES),
        "status": status,
        "delivered_at": None,
        "delivery_fee_cents": fees[0],
        "service_fee_cents": fees[1],
        "discount_cents": discount,
        "total_cents": subtotal + sum(fees) - discount,
        "observacao": rng.choice(NOTES) if rng.random() < NOTE_PROBABILITY else "",
        "items": items,
        **_timeline(rng, now, status, edge[1] if edge else None),
    }


def generate(
    conn: sqlite3.Connection,
    seed: int = 7,
    n_customers: int = 30,
    n_orders: int = 200,
    now: datetime = REFERENCE_NOW,
    extra_orders: list[dict] | None = None,
) -> None:
    rng = random.Random(seed)
    customers = [
        (f"C{i:03d}", f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}")
        for i in range(1, n_customers + 1)
    ]
    conn.executemany("INSERT INTO customers (id, nome) VALUES (?, ?)", customers)
    edges = _edge_slots(rng)
    for k, n in enumerate(rng.sample(range(*ORDER_ID_RANGE), n_orders)):
        edge = edges[k] if k < len(edges) else None
        order = _build_order(rng, f"P{n:05d}", rng.choice(customers)[0], now, edge)
        _insert_order(conn, order)
    known = {c[0] for c in customers}
    for order in extra_orders or []:
        if order["customer_id"] not in known:
            raise ValueError(f"unknown customer_id: {order['customer_id']}")
        _insert_order(conn, order)
    conn.commit()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m support_agent.orders")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="create a database with the default synthetic orders")
    gen.add_argument("db_path", type=Path)
    gen.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args(argv)
    if args.db_path.exists():
        if not args.force:
            parser.error(f"{args.db_path} exists; pass --force to overwrite")
        args.db_path.unlink()
    conn = connect(args.db_path)
    generate(conn)
    conn.close()


if __name__ == "__main__":
    main()
