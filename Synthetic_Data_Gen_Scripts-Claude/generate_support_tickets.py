"""
generate_support_tickets.py
Generates synthetic customer support ticket data and saves to support_tickets.csv
Depends on customers.csv and orders.csv being present.
"""

import csv
import random
import uuid
from datetime import datetime, timedelta

# --- Config ---
NUM_TICKETS = 1000
OUTPUT_FILE = "support_tickets.csv"
CUSTOMERS_FILE = "customers.csv"
ORDERS_FILE = "orders.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --- Data pools ---
ISSUE_CATEGORIES = {
    "Order & Shipping": [
        "Order not received", "Wrong item delivered", "Damaged product",
        "Delayed shipment", "Tracking not updating", "Order cancelled by mistake",
    ],
    "Payment & Billing": [
        "Double charged", "Refund not received", "Payment failed", 
        "Incorrect invoice", "Coupon not applied", "EMI issue",
    ],
    "Product Quality": [
        "Product defective", "Product not as described", "Missing accessories",
        "Product stopped working", "Size/color mismatch",
    ],
    "Returns & Refunds": [
        "Want to return item", "Refund delayed", "Return pickup not scheduled",
        "Partial refund received", "Return rejected",
    ],
    "Account & Login": [
        "Cannot login", "Password reset issue", "Account hacked", 
        "Email not verified", "Profile update error",
    ],
    "Technical Issues": [
        "App crashing", "Website not loading", "Search not working",
        "Checkout error", "Notification issue",
    ],
}

PRIORITIES = ["Low", "Medium", "High", "Critical"]
PRIORITY_WEIGHTS = [0.3, 0.4, 0.2, 0.1]

STATUSES = ["Open", "In Progress", "Pending Customer", "Resolved", "Closed", "Escalated"]
STATUS_WEIGHTS = [0.10, 0.15, 0.08, 0.45, 0.17, 0.05]

CHANNELS = ["Email", "Chat", "Phone", "Social Media", "In-App", "Web Form"]
CHANNEL_WEIGHTS = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]

AGENTS = [
    "agent_alice", "agent_bob", "agent_carlos", "agent_diana",
    "agent_evan", "agent_fiona", "agent_george", "agent_hannah",
]

SENTIMENT = ["Positive", "Neutral", "Negative", "Very Negative"]
SENTIMENT_WEIGHTS = [0.15, 0.30, 0.35, 0.20]

RESOLUTION_NOTES = [
    "Issue resolved after investigation.", "Refund processed successfully.",
    "Replacement dispatched.", "Account issue fixed by tech team.",
    "Customer informed of policy.", "Escalated to senior support.",
    "No action required — user error.", "Vendor contacted for resolution.",
    "Compensation coupon issued.", "Issue reproduced and logged as bug.",
]


def load_ids(filepath, col):
    ids = []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row[col])
    return ids


def random_date(start_year=2021, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_ticket(customer_ids, order_ids):
    category = random.choice(list(ISSUE_CATEGORIES.keys()))
    issue = random.choice(ISSUE_CATEGORIES[category])
    status = random.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
    created_at = random_date()
    priority = random.choices(PRIORITIES, weights=PRIORITY_WEIGHTS)[0]

    # Resolution time depends on priority
    resolution_hours = {
        "Critical": random.randint(1, 12),
        "High": random.randint(4, 48),
        "Medium": random.randint(12, 96),
        "Low": random.randint(24, 168),
    }
    resolved_at = (
        created_at + timedelta(hours=resolution_hours[priority])
        if status in ("Resolved", "Closed")
        else None
    )

    first_response_minutes = random.randint(2, 120)

    return {
        "ticket_id": f"TKT-{random.randint(10000, 99999)}",
        "customer_id": random.choice(customer_ids),
        "order_id": random.choice(order_ids) if random.random() < 0.7 else "",
        "category": category,
        "issue_description": issue,
        "channel": random.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0],
        "priority": priority,
        "status": status,
        "assigned_agent": random.choice(AGENTS),
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": (created_at + timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%d %H:%M:%S"),
        "resolved_at": resolved_at.strftime("%Y-%m-%d %H:%M:%S") if resolved_at else "",
        "first_response_minutes": first_response_minutes,
        "resolution_time_hours": round(resolution_hours[priority], 1) if resolved_at else "",
        "customer_sentiment": random.choices(SENTIMENT, weights=SENTIMENT_WEIGHTS)[0],
        "csat_score": random.choice(["", "1", "2", "3", "4", "5"]) if resolved_at else "",
        "resolution_note": random.choice(RESOLUTION_NOTES) if resolved_at else "",
        "is_escalated": status == "Escalated",
        "tags": random.choice(["", "vip", "repeat-issue", "bug", "refund", "urgent", "follow-up"]),
    }


def main():
    try:
        customer_ids = load_ids(CUSTOMERS_FILE, "customer_id")
        order_ids = load_ids(ORDERS_FILE, "order_id")
    except FileNotFoundError as e:
        print(f"⚠️  Dependency missing: {e}")
        print("   Run generate_customers.py and generate_orders.py first.")
        raise

    tickets = [generate_ticket(customer_ids, order_ids) for _ in range(NUM_TICKETS)]
    fields = list(tickets[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tickets)

    print(f"✅ Generated {NUM_TICKETS} support tickets → {OUTPUT_FILE}")
    return tickets


if __name__ == "__main__":
    main()
