"""Business rules of the fictional delivery platform: single source for tools and tests.

The policy Markdown in corpus/policies/ states the same values in prose; POLICY_SOURCES
links each constant to the passage that must mention it, so text and code cannot drift.
"""

STORE_TYPES = ("restaurante", "mercado", "varejo")
# "propria": the platform's couriers deliver; "parceiro": the store delivers itself.
DELIVERY_MODES = ("propria", "parceiro")

# Days after delivered_at to request a refund for a missing/wrong item or quality issue.
COMPLAINT_WINDOW_DAYS: dict[str, int] = {"restaurante": 2, "mercado": 7, "varejo": 30}

# Regret right (CDC art. 49), counted from delivered_at. Restaurants are excluded because
# their food is freshly prepared and perishable.
REGRET_WINDOW_DAYS = 7
REGRET_STORE_TYPES = ("mercado", "varejo")

REFUND_REASONS = ("item_faltando", "item_errado", "qualidade", "nao_entregue", "arrependimento")

# A restaurant order with platform delivery ("propria") later than this earns a coupon,
# not a refund.
LATE_COUPON_MINUTES = 30
LATE_COUPON_CENTS = 1000

# Constant name (dotted key for dict entries) -> passage_id whose text states the value.
POLICY_SOURCES: dict[str, str] = {
    "COMPLAINT_WINDOW_DAYS.restaurante": "reembolso_restaurante:2.1",
    "COMPLAINT_WINDOW_DAYS.mercado": "reembolso_mercado_varejo:2.1",
    "COMPLAINT_WINDOW_DAYS.varejo": "reembolso_mercado_varejo:2.2",
    "REGRET_WINDOW_DAYS": "reembolso_mercado_varejo:3.1",
    "REGRET_STORE_TYPES": "reembolso_mercado_varejo:3.1",
    "LATE_COUPON_MINUTES": "atrasos:2",
    "LATE_COUPON_CENTS": "atrasos:2",
}
