# Script 3: SQL Queries & Window Functions
# ==========================================

# Create sample data
sales_data = [
    (1, "Product A", "North", 100, 50.0),
    (2, "Product B", "South", 150, 75.0),
    (1, "Product A", "North", 120, 60.0),
    (3, "Product C", "East", 200, 100.0),
    (2, "Product B", "South", 180, 90.0),
    (1, "Product A", "North", 110, 55.0),
]

df = spark.createDataFrame(
    sales_data,
    ["transaction_id", "product_name", "region", "quantity", "price"]
)

# Create temporary view for SQL queries
df.createOrReplaceTempView("sales")

# 1. BASIC SQL SELECT
print("=== Basic SELECT ===")
result = spark.sql("""
    SELECT 
        product_name, 
        region, 
        quantity * price as revenue
    FROM sales
    WHERE quantity > 100
    ORDER BY revenue DESC
""")
result.show()
print()

# 2. GROUP BY & AGGREGATION
print("=== GROUP BY & Aggregation ===")
result = spark.sql("""
    SELECT 
        region,
        COUNT(*) as num_transactions,
        SUM(quantity) as total_quantity,
        SUM(quantity * price) as total_revenue,
        AVG(price) as avg_price,
        MIN(price) as min_price,
        MAX(price) as max_price
    FROM sales
    GROUP BY region
    ORDER BY total_revenue DESC
""")
result.show()
print()

# 3. HAVING CLAUSE (filter aggregates)
print("=== HAVING Clause ===")
result = spark.sql("""
    SELECT 
        product_name,
        COUNT(*) as count,
        SUM(quantity * price) as total_revenue
    FROM sales
    GROUP BY product_name
    HAVING SUM(quantity * price) > 500
""")
result.show()
print()

# 4. WINDOW FUNCTIONS - ROW_NUMBER
print("=== Window Function: ROW_NUMBER ===")
result = spark.sql("""
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY region ORDER BY price DESC) as rank_in_region
    FROM sales
""")
result.show()
print()

# 5. WINDOW FUNCTIONS - RANK
print("=== Window Function: RANK ===")
result = spark.sql("""
    SELECT 
        *,
        RANK() OVER (ORDER BY price DESC) as overall_rank
    FROM sales
""")
result.show()
print()

# 6. WINDOW FUNCTIONS - Running Total
print("=== Window Function: Running Total ===")
result = spark.sql("""
    SELECT 
        transaction_id,
        product_name,
        price,
        SUM(price) OVER (ORDER BY transaction_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total
    FROM sales
""")
result.show()
print()

# 7. WINDOW FUNCTIONS - LAG & LEAD
print("=== Window Function: LAG & LEAD ===")
result = spark.sql("""
    SELECT 
        transaction_id,
        product_name,
        price,
        LAG(price) OVER (ORDER BY transaction_id) as prev_price,
        LEAD(price) OVER (ORDER BY transaction_id) as next_price
    FROM sales
""")
result.show()
print()

# 8. CTE (Common Table Expression)
print("=== CTE Example ===")
result = spark.sql("""
    WITH regional_totals AS (
        SELECT 
            region,
            SUM(quantity * price) as total_revenue
        FROM sales
        GROUP BY region
    )
    SELECT 
        region,
        total_revenue,
        CASE 
            WHEN total_revenue > 1000 THEN 'High'
            WHEN total_revenue > 500 THEN 'Medium'
            ELSE 'Low'
        END as revenue_category
    FROM regional_totals
""")
result.show()
print()

# 9. JOIN (create second table for demo)
print("=== JOIN Example ===")

# Create products lookup table
products_data = [
    ("Product A", "Low Cost"),
    ("Product B", "Mid Cost"),
    ("Product C", "Premium"),
]
df_products = spark.createDataFrame(
    products_data,
    ["product_name", "category"]
)
df_products.createOrReplaceTempView("products")

# Perform JOIN
result = spark.sql("""
    SELECT 
        s.transaction_id,
        s.product_name,
        p.category,
        s.region,
        s.quantity * s.price as revenue
    FROM sales s
    INNER JOIN products p ON s.product_name = p.product_name
    ORDER BY s.transaction_id
""")
result.show()
print()

# 10. SUBQUERY
print("=== Subquery Example ===")
result = spark.sql("""
    SELECT 
        product_name,
        SUM(quantity * price) as total_revenue
    FROM sales
    WHERE region IN (
        SELECT region FROM sales 
        WHERE quantity > 150
    )
    GROUP BY product_name
""")
result.show()
print()

# 11. UNION (combine result sets)
print("=== UNION Example ===")
result = spark.sql("""
    SELECT region, COUNT(*) as count, 'Sales' as source FROM sales
    WHERE quantity > 100
    GROUP BY region
    
    UNION
    
    SELECT region, 1 as count, 'New Region' as source FROM sales
    WHERE quantity < 100
    GROUP BY region
""")
result.show()
