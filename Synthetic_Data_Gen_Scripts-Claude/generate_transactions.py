"""
generate_transactions.py
Generates synthetic financial transactions tied to orders and saves to transactions.csv
Depends on orders.csv being present.
"""

import csv
import random
import uuid
from datetime import datetime, timedelta

# --- Config ---
OUTPUT_FILE = "transactions.csv"
ORDERS_FILE = "orders.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --- Data pools ---
TRANSACTION_TYPES = ["Payment", "Refund", "Chargeback", "Adjustment"]

PAYMENT_GATEWAYS = ["Stripe", "PayPal", "Razorpay", "Square", "Adyen", "Braintree", "Paytm", "PhonePe"]

CURRENCIES = ["USD", "USD", "USD", "INR", "INR", "GBP", "EUR", "CAD", "AUD", "JPY"]

STATUSES = ["Success", "Success", "Success", "Success", "Failed", "Pending", "Reversed"]

# Approximate FX rate to USD (1 unit of currency = X USD)
FX_RATES = {
    "USD": 1.0, "INR": 0.012, "GBP": 1.27, "EUR": 1.09, "CAD": 0.74, "AUD": 0.65, "JPY": 0.0067
}


def load_orders(filepath):
    orders = []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orders.append({
                "order_id": row["order_id"],
                "customer_id": row["customer_id"],
                "order_date": row["order_date"],
                "total_amount": float(row["total_amount"]),
                "payment_method": row["payment_method"],
            })
    return orders


def generate_transaction(order):
    currency = random.choice(CURRENCIES)
    fx = FX_RATES[currency]
    amount_local = round(order["total_amount"] / fx, 2)
    tx_type = random.choices(
        TRANSACTION_TYPES,
        weights=[0.80, 0.12, 0.04, 0.04]
    )[0]

    # Refunds/chargebacks are partial or full
    if tx_type in ("Refund", "Chargeback"):
        amount_local = round(amount_local * random.uniform(0.3, 1.0), 2)

    status = random.choice(STATUSES)

    # Transaction timestamp: same day as order or +1 day
    order_dt = datetime.strptime(order["order_date"], "%Y-%m-%d")
    tx_dt = order_dt + timedelta(
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    settled_dt = tx_dt + timedelta(days=random.randint(1, 3)) if status == "Success" else None

    # Fee charged by payment gateway (0.5% – 3%)
    fee_pct = random.uniform(0.005, 0.03)
    gateway_fee = round(amount_local * fee_pct, 2)

    return {
        "transaction_id": str(uuid.uuid4()),
        "order_id": order["order_id"],
        "customer_id": order["customer_id"],
        "transaction_type": tx_type,
        "payment_method": order["payment_method"],
        "payment_gateway": random.choice(PAYMENT_GATEWAYS),
        "currency": currency,
        "amount": amount_local,
        "amount_usd": round(amount_local * fx, 2),
        "gateway_fee": gateway_fee,
        "net_amount": round(amount_local - gateway_fee, 2),
        "status": status,
        "transaction_datetime": tx_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "settled_datetime": settled_dt.strftime("%Y-%m-%d %H:%M:%S") if settled_dt else "",
        "reference_code": f"REF-{random.randint(100000, 999999)}",
        "ip_country": random.choice(["US", "IN", "GB", "CA", "AU", "DE", "JP", "SG"]),
        "is_flagged": random.choices([True, False], weights=[0.03, 0.97])[0],
    }


def main():
    try:
        orders = load_orders(ORDERS_FILE)
    except FileNotFoundError as e:
        print(f"⚠️  Dependency missing: {e}")
        print("   Run generate_orders.py first.")
        raise

    transactions = [generate_transaction(order) for order in orders]
    fields = list(transactions[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(transactions)

    print(f"✅ Generated {len(transactions)} transactions → {OUTPUT_FILE}")
    return transactions


if __name__ == "__main__":
    main()
