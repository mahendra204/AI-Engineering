"""
generate_orders.py
Generates synthetic orders data and saves to orders.csv
Depends on customers.csv and products.csv being present (or re-generates them).
"""

import csv
import random
import uuid
from datetime import datetime, timedelta

# --- Config ---
NUM_ORDERS = 2000
OUTPUT_FILE = "orders.csv"
CUSTOMERS_FILE = "customers.csv"
PRODUCTS_FILE = "products.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --- Lookup helpers ---
def load_ids(filepath, id_col):
    ids = []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row[id_col])
    return ids


def load_products(filepath):
    products = []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            products.append({
                "product_id": row["product_id"],
                "final_price": float(row["final_price"]),
                "category": row["category"],
            })
    return products


# --- Data pools ---
ORDER_STATUSES = ["Pending", "Processing", "Shipped", "Delivered", "Cancelled", "Returned"]
STATUS_WEIGHTS  = [0.05, 0.10, 0.15, 0.55, 0.10, 0.05]

PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet", "Cash on Delivery", "EMI"]
PAYMENT_WEIGHTS  = [0.25, 0.20, 0.25, 0.08, 0.10, 0.07, 0.05]

SHIPPING_METHODS = ["Standard", "Express", "Same Day", "Pickup"]
SHIPPING_WEIGHTS  = [0.55, 0.30, 0.10, 0.05]

SHIPPING_COSTS = {"Standard": 0, "Express": 4.99, "Same Day": 9.99, "Pickup": 0}

COUPON_CODES = [None, None, None, None, "SAVE10", "FLAT20", "WELCOME15", "SUMMER25", "FLASH30", "LOYAL5"]


def random_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_order(customer_ids, products):
    order_date = random_date()
    status = random.choices(ORDER_STATUSES, weights=STATUS_WEIGHTS)[0]
    ship_method = random.choices(SHIPPING_METHODS, weights=SHIPPING_WEIGHTS)[0]

    # Pick 1–5 random products (simulate line items counted in orders)
    num_items = random.randint(1, 5)
    chosen = random.choices(products, k=num_items)
    subtotal = round(sum(p["final_price"] * random.randint(1, 3) for p in chosen), 2)
    shipping_cost = SHIPPING_COSTS[ship_method]
    tax = round(subtotal * 0.08, 2)
    discount = round(subtotal * random.uniform(0, 0.2), 2) if random.random() < 0.3 else 0
    total = round(subtotal + shipping_cost + tax - discount, 2)

    coupon = random.choice(COUPON_CODES)

    # Estimated delivery date
    delivery_days = {"Standard": 5, "Express": 2, "Same Day": 0, "Pickup": 0}
    est_delivery = order_date + timedelta(days=delivery_days[ship_method])
    actual_delivery = est_delivery + timedelta(days=random.randint(-1, 3)) if status == "Delivered" else None

    return {
        "order_id": str(uuid.uuid4()),
        "customer_id": random.choice(customer_ids),
        "order_date": order_date.date(),
        "status": status,
        "payment_method": random.choices(PAYMENT_METHODS, weights=PAYMENT_WEIGHTS)[0],
        "shipping_method": ship_method,
        "shipping_cost": shipping_cost,
        "subtotal": subtotal,
        "tax": tax,
        "discount": discount,
        "coupon_code": coupon if coupon else "",
        "total_amount": total,
        "num_items": num_items,
        "estimated_delivery": est_delivery.date(),
        "actual_delivery": actual_delivery.date() if actual_delivery else "",
        "shipping_address_city": random.choice([
            "New York", "Los Angeles", "Mumbai", "Bengaluru", "London",
            "Toronto", "Sydney", "Tokyo", "Berlin", "Singapore"
        ]),
        "is_gift": random.choices([True, False], weights=[0.1, 0.9])[0],
        "feedback_rating": random.choice(["", "", "1", "2", "3", "4", "5"]),
    }


def main():
    # Load dependencies
    try:
        customer_ids = load_ids(CUSTOMERS_FILE, "customer_id")
        products = load_products(PRODUCTS_FILE)
    except FileNotFoundError as e:
        print(f"⚠️  Dependency missing: {e}")
        print("   Run generate_customers.py and generate_products.py first.")
        raise

    orders = [generate_order(customer_ids, products) for _ in range(NUM_ORDERS)]
    fields = list(orders[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(orders)

    print(f"✅ Generated {NUM_ORDERS} orders → {OUTPUT_FILE}")
    return orders


if __name__ == "__main__":
    main()
