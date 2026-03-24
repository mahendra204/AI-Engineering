"""
generate_products.py
Generates synthetic product catalog and saves to products.csv
"""

import csv
import random
import uuid

# --- Config ---
NUM_PRODUCTS = 200
OUTPUT_FILE = "products.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --- Data pools ---
CATEGORIES = {
    "Electronics": {
        "subcategories": ["Smartphones", "Laptops", "Tablets", "Headphones", "Cameras", "Smartwatches"],
        "brands": ["Apple", "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo", "Bose"],
        "price_range": (50, 3000),
    },
    "Clothing": {
        "subcategories": ["Men's Wear", "Women's Wear", "Kids", "Sportswear", "Footwear", "Accessories"],
        "brands": ["Nike", "Adidas", "Zara", "H&M", "Levi's", "Puma", "Gucci", "Uniqlo"],
        "price_range": (10, 500),
    },
    "Home & Kitchen": {
        "subcategories": ["Cookware", "Furniture", "Decor", "Appliances", "Bedding", "Lighting"],
        "brands": ["IKEA", "Dyson", "KitchenAid", "Philips", "Whirlpool", "Prestige"],
        "price_range": (15, 2000),
    },
    "Books": {
        "subcategories": ["Fiction", "Non-Fiction", "Science", "Technology", "Self-Help", "Children's"],
        "brands": ["Penguin", "HarperCollins", "O'Reilly", "Scholastic", "Oxford Press"],
        "price_range": (5, 80),
    },
    "Sports & Fitness": {
        "subcategories": ["Gym Equipment", "Outdoor Gear", "Cycling", "Yoga", "Swimming", "Team Sports"],
        "brands": ["Nike", "Adidas", "Decathlon", "Reebok", "Under Armour", "Yonex"],
        "price_range": (10, 1500),
    },
    "Beauty & Personal Care": {
        "subcategories": ["Skincare", "Haircare", "Makeup", "Fragrances", "Grooming"],
        "brands": ["L'Oreal", "Nivea", "Dove", "MAC", "The Body Shop", "Mamaearth"],
        "price_range": (5, 300),
    },
    "Groceries": {
        "subcategories": ["Snacks", "Beverages", "Dairy", "Grains", "Organic", "Frozen Food"],
        "brands": ["Nestlé", "Kellogs", "Amul", "Britannia", "PepsiCo", "Organic India"],
        "price_range": (1, 50),
    },
}

ADJECTIVES = ["Premium", "Ultra", "Pro", "Plus", "Max", "Elite", "Classic", "Smart", "Eco", "Lite"]
NOUNS = ["Series", "Edition", "Collection", "Pack", "Bundle", "Kit", "Set", "Version", "Model", "Line"]


def generate_product_name(subcategory, brand):
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    number = random.choice(["", f" {random.randint(1,9)}00", f" {random.randint(1,20)}"])
    return f"{brand} {subcategory} {adj} {noun}{number}".strip()


def generate_product(category_name, category_data):
    subcategory = random.choice(category_data["subcategories"])
    brand = random.choice(category_data["brands"])
    low, high = category_data["price_range"]
    cost_price = round(random.uniform(low * 0.4, high * 0.6), 2)
    selling_price = round(cost_price * random.uniform(1.2, 2.5), 2)
    discount_pct = random.choice([0, 0, 0, 5, 10, 15, 20, 25, 30])

    return {
        "product_id": str(uuid.uuid4()),
        "product_name": generate_product_name(subcategory, brand),
        "category": category_name,
        "subcategory": subcategory,
        "brand": brand,
        "sku": f"SKU-{random.randint(10000, 99999)}",
        "cost_price": cost_price,
        "selling_price": selling_price,
        "discount_percent": discount_pct,
        "final_price": round(selling_price * (1 - discount_pct / 100), 2),
        "stock_quantity": random.randint(0, 1000),
        "is_available": random.choices([True, False], weights=[0.9, 0.1])[0],
        "rating": round(random.uniform(2.5, 5.0), 1),
        "review_count": random.randint(0, 5000),
        "weight_kg": round(random.uniform(0.1, 20.0), 2),
        "supplier_country": random.choice(["China", "India", "USA", "Germany", "Japan", "South Korea"]),
    }


def main():
    products = []
    category_names = list(CATEGORIES.keys())

    for _ in range(NUM_PRODUCTS):
        cat_name = random.choice(category_names)
        products.append(generate_product(cat_name, CATEGORIES[cat_name]))

    fields = list(products[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(products)

    print(f"✅ Generated {NUM_PRODUCTS} products → {OUTPUT_FILE}")
    return products


if __name__ == "__main__":
    main()
