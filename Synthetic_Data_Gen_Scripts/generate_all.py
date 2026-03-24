"""
generate_all.py
Master runner — generates all synthetic datasets in the correct dependency order.

Run:  python generate_all.py
Output files (in the same directory):
  customers.csv
  products.csv
  orders.csv
  transactions.csv
  employees.csv
  support_tickets.csv
"""

import os
import sys

# Change working directory to the script's location so all CSVs land here.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  Synthetic Data Generator — All Datasets")
print("=" * 55)

# 1. Customers (no dependencies)
print("\n[1/6] Customers")
from generate_customers import main as gen_customers
gen_customers()

# 2. Products (no dependencies)
print("\n[2/6] Products")
from generate_products import main as gen_products
gen_products()

# 3. Orders (needs customers + products)
print("\n[3/6] Orders")
from generate_orders import main as gen_orders
gen_orders()

# 4. Transactions (needs orders)
print("\n[4/6] Transactions")
from generate_transactions import main as gen_transactions
gen_transactions()

# 5. Employees (no dependencies)
print("\n[5/6] Employees")
from generate_employees import main as gen_employees
gen_employees()

# 6. Support Tickets (needs customers + orders)
print("\n[6/6] Support Tickets")
from generate_support_tickets import main as gen_support
gen_support()

print("\n" + "=" * 55)
print("  ✅ All datasets generated successfully!")
print("=" * 55)

# Print summary
import csv

files = [
    "customers.csv", "products.csv", "orders.csv",
    "transactions.csv", "employees.csv", "support_tickets.csv"
]

print(f"\n{'File':<25} {'Rows':>8}  {'Size':>10}")
print("-" * 47)
for fname in files:
    if os.path.exists(fname):
        with open(fname, newline="") as f:
            rows = sum(1 for _ in f) - 1  # subtract header
        size_kb = os.path.getsize(fname) / 1024
        print(f"{fname:<25} {rows:>8,}  {size_kb:>8.1f} KB")

print()
