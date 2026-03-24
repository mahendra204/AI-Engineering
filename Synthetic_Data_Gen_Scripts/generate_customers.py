"""
generate_customers.py
Generates synthetic customer data and saves to customers.csv
"""

import csv
import random
import uuid
from datetime import datetime, timedelta

# --- Config ---
NUM_CUSTOMERS = 500
OUTPUT_FILE = "customers.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --- Data pools ---
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Priya", "Arjun", "Sneha", "Rahul",
    "Ananya", "Vikram", "Deepa", "Rohan", "Meera", "Karan", "Aisha", "Mohammed",
    "Fatima", "Omar", "Layla", "Hassan", "Yuki", "Kenji", "Sakura", "Takeshi",
    "Elena", "Dmitri", "Sofia", "Lucas", "Isabella", "Mateo", "Valentina", "Santiago"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Patel", "Sharma", "Singh", "Kumar", "Shah", "Mehta", "Gupta", "Reddy",
    "Khan", "Ali", "Ahmed", "Hassan", "Yamamoto", "Tanaka", "Watanabe", "Suzuki",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Taylor", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Moore"
]

DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "protonmail.com", "zoho.com", "rediffmail.com", "company.com", "business.org"
]

CITIES = [
    ("New York", "NY", "USA"), ("Los Angeles", "CA", "USA"), ("Chicago", "IL", "USA"),
    ("Houston", "TX", "USA"), ("Phoenix", "AZ", "USA"), ("London", "", "UK"),
    ("Manchester", "", "UK"), ("Mumbai", "MH", "India"), ("Delhi", "DL", "India"),
    ("Bengaluru", "KA", "India"), ("Hyderabad", "TS", "India"), ("Chennai", "TN", "India"),
    ("Toronto", "ON", "Canada"), ("Vancouver", "BC", "Canada"), ("Sydney", "NSW", "Australia"),
    ("Melbourne", "VIC", "Australia"), ("Berlin", "", "Germany"), ("Paris", "", "France"),
    ("Tokyo", "", "Japan"), ("Singapore", "", "Singapore")
]

SEGMENTS = ["Standard", "Premium", "VIP", "Enterprise"]
SEGMENT_WEIGHTS = [0.5, 0.3, 0.15, 0.05]

ACQUISITION_CHANNELS = ["Organic Search", "Paid Ads", "Referral", "Social Media", "Email Campaign", "Direct"]


def random_date(start_year=2018, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).date()


def generate_phone():
    return f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"


def generate_customer():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    email_prefix = f"{first.lower()}.{last.lower()}{random.randint(1, 999)}"
    email = f"{email_prefix}@{random.choice(DOMAINS)}"
    city, state, country = random.choice(CITIES)
    signup_date = random_date()
    segment = random.choices(SEGMENTS, weights=SEGMENT_WEIGHTS)[0]
    is_active = random.choices([True, False], weights=[0.85, 0.15])[0]

    return {
        "customer_id": str(uuid.uuid4()),
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": generate_phone(),
        "city": city,
        "state": state,
        "country": country,
        "signup_date": signup_date,
        "segment": segment,
        "acquisition_channel": random.choice(ACQUISITION_CHANNELS),
        "is_active": is_active,
        "date_of_birth": random_date(1960, 2003),
        "loyalty_points": random.randint(0, 50000) if segment != "Standard" else random.randint(0, 5000),
    }


def main():
    customers = [generate_customer() for _ in range(NUM_CUSTOMERS)]
    fields = list(customers[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(customers)

    print(f"✅ Generated {NUM_CUSTOMERS} customers → {OUTPUT_FILE}")
    return customers  # returned for use by other generators


if __name__ == "__main__":
    main()
