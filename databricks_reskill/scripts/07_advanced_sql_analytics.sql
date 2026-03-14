# SQL Script: Comprehensive Analytics Queries
# ============================================

-- ===== SALES ANALYTICS DATABASE SETUP =====

-- Create sample tables
CREATE TABLE IF NOT EXISTS sales_data (
    order_id INT,
    customer_id INT,
    product_id INT,
    order_date DATE,
    amount DOUBLE,
    quantity INT,
    region STRING
)
USING DELTA;

CREATE TABLE IF NOT EXISTS customers (
    customer_id INT,
    name STRING,
    email STRING,
    country STRING,
    signup_date DATE
)
USING DELTA;

CREATE TABLE IF NOT EXISTS products (
    product_id INT,
    product_name STRING,
    category STRING,
    price DOUBLE
)
USING DELTA;

-- ===== BASIC ANALYTICS =====

-- 1. Total sales by region
SELECT
    region,
    COUNT(*) as num_orders,
    SUM(amount) as total_sales,
    AVG(amount) as avg_order_value,
    MIN(amount) as min_sale,
    MAX(amount) as max_sale
FROM sales_data
GROUP BY region
ORDER BY total_sales DESC;

-- 2. Sales trends over time
SELECT
    DATE_TRUNC('month', order_date) as month,
    COUNT(*) as orders,
    SUM(amount) as sales,
    AVG(amount) as avg_amount
FROM sales_data
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month DESC;

-- 3. Top 10 customers by revenue
SELECT
    c.customer_id,
    c.name,
    COUNT(s.order_id) as num_orders,
    SUM(s.amount) as total_spent,
    AVG(s.amount) as avg_order
FROM customers c
LEFT JOIN sales_data s ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC
LIMIT 10;

-- ===== WINDOW FUNCTIONS =====

-- 4. Rank products by revenue
SELECT
    product_id,
    product_name,
    SUM(sales.amount) as revenue,
    RANK() OVER (ORDER BY SUM(sales.amount) DESC) as rank
FROM products p
LEFT JOIN sales_data sales ON p.product_id = sales.product_id
GROUP BY p.product_id, p.product_name
ORDER BY rank;

-- 5. Running total of sales
SELECT
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total,
    ROW_NUMBER() OVER (ORDER BY order_date) as row_num
FROM sales_data
ORDER BY order_date;

-- 6. Rank customers within each region
SELECT
    s.customer_id,
    c.name,
    s.region,
    SUM(s.amount) as regional_sales,
    RANK() OVER (PARTITION BY s.region ORDER BY SUM(s.amount) DESC) as rank_in_region
FROM sales_data s
JOIN customers c ON s.customer_id = c.customer_id
GROUP BY s.customer_id, c.name, s.region
ORDER BY s.region, rank_in_region;

-- 7. Moving average (30-day window)
SELECT
    order_date,
    amount,
    AVG(amount) OVER (ORDER BY order_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as moving_avg_30d
FROM sales_data
ORDER BY order_date;

-- 8. Year-over-year comparison
SELECT
    YEAR(order_date) as year,
    MONTH(order_date) as month,
    SUM(amount) as sales,
    LAG(SUM(amount)) OVER (PARTITION BY MONTH(order_date) ORDER BY YEAR(order_date)) as prev_year_sales
FROM sales_data
GROUP BY YEAR(order_date), MONTH(order_date)
ORDER BY year DESC, month;

-- ===== COHORT ANALYSIS =====

-- 9. Customer cohort analysis
WITH customer_cohorts AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) as cohort_month,
        SUM(amount) as total_spent,
        COUNT(*) as num_orders
    FROM sales_data
    GROUP BY customer_id
)
SELECT
    cohort_month,
    COUNT(*) as customers_acquired,
    AVG(total_spent) as avg_ltv,
    SUM(total_spent) as cohort_revenue
FROM customer_cohorts
GROUP BY cohort_month
ORDER BY cohort_month DESC;

-- ===== SEGMENTATION =====

-- 10. Customer segmentation by behavior
SELECT
    customer_id,
    CASE
        WHEN total_spent > 1000 THEN 'VIP'
        WHEN total_spent > 500 THEN 'Premium'
        WHEN total_spent > 100 THEN 'Regular'
        ELSE 'New'
    END as segment,
    num_orders,
    total_spent,
    CASE
        WHEN days_since_order <= 30 THEN 'Active'
        WHEN days_since_order <= 60 THEN 'At Risk'
        ELSE 'Inactive'
    END as status
FROM (
    SELECT
        customer_id,
        SUM(amount) as total_spent,
        COUNT(*) as num_orders,
        DATEDIFF(CURRENT_DATE(), MAX(order_date)) as days_since_order
    FROM sales_data
    GROUP BY customer_id
);

-- ===== PERFORMANCE METRICS =====

-- 11. Key performance indicators
SELECT
    CURRENT_DATE() as report_date,
    COUNT(DISTINCT customer_id) as unique_customers,
    COUNT(*) as total_orders,
    SUM(amount) as total_revenue,
    AVG(amount) as avg_order_value,
    MAX(amount) as max_order,
    STDDEV(amount) as revenue_std_dev,
    COUNT(DISTINCT region) as regions_served
FROM sales_data;

-- 12. Product performance
SELECT
    p.product_name,
    p.category,
    COUNT(s.order_id) as times_sold,
    SUM(s.quantity) as units_sold,
    SUM(s.amount) as total_revenue,
    AVG(s.amount) as avg_sale,
    MIN(s.order_date) as first_sale,
    MAX(s.order_date) as last_sale
FROM products p
LEFT JOIN sales_data s ON p.product_id = s.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_revenue DESC;

-- ===== COMPLEX JOINS & SUBQUERIES =====

-- 13. Customers who bought in multiple regions
SELECT
    c.customer_id,
    c.name,
    COUNT(DISTINCT s.region) as regions_purchased,
    STRING_AGG(DISTINCT s.region, ', ') as regions_list,
    SUM(s.amount) as total_spent
FROM customers c
JOIN sales_data s ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.name
HAVING COUNT(DISTINCT s.region) > 1
ORDER BY total_spent DESC;

-- 14. Products never sold in specific region
SELECT
    p.product_id,
    p.product_name,
    COUNT(DISTINCT r.region) as regions_not_sold
FROM products p
CROSS JOIN (SELECT DISTINCT region FROM sales_data) r
WHERE p.product_id NOT IN (
    SELECT DISTINCT product_id FROM sales_data WHERE region = r.region
)
GROUP BY p.product_id, p.product_name;

-- ===== ANOMALY DETECTION =====

-- 15. Orders significantly above average
WITH avg_calc AS (
    SELECT
        AVG(amount) as avg_amount,
        STDDEV(amount) as std_dev
    FROM sales_data
)
SELECT
    s.order_id,
    s.customer_id,
    s.amount,
    ROUND((s.amount - a.avg_amount) / a.std_dev, 2) as z_score,
    s.order_date
FROM sales_data s
CROSS JOIN avg_calc a
WHERE s.amount > (a.avg_amount + 2 * a.std_dev)
ORDER BY s.amount DESC;

-- ===== COMMON TABLE EXPRESSIONS (CTEs) =====

-- 16. Multi-level CTE analysis
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', order_date) as month,
        region,
        SUM(amount) as monthly_revenue
    FROM sales_data
    GROUP BY DATE_TRUNC('month', order_date), region
),
regional_avg AS (
    SELECT
        region,
        AVG(monthly_revenue) as avg_monthly_revenue
    FROM monthly_sales
    GROUP BY region
),
variance_calc AS (
    SELECT
        ms.month,
        ms.region,
        ms.monthly_revenue,
        ra.avg_monthly_revenue,
        ROUND(((ms.monthly_revenue - ra.avg_monthly_revenue) / ra.avg_monthly_revenue * 100), 2) as variance_percent
    FROM monthly_sales ms
    JOIN regional_avg ra ON ms.region = ra.region
)
SELECT
    *
FROM variance_calc
WHERE ABS(variance_percent) > 20
ORDER BY month DESC, variance_percent DESC;

-- ===== CUMULATIVE METRICS =====

-- 17. Cumulative revenue by customer
SELECT
    customer_id,
    order_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_revenue,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_number
FROM sales_data
ORDER BY customer_id, order_date;

-- ===== EXPORT RESULTS =====

-- Save top customers to Delta table
CREATE OR REPLACE TABLE top_customers AS
SELECT
    c.customer_id,
    c.name,
    c.email,
    COUNT(s.order_id) as num_orders,
    SUM(s.amount) as total_spent,
    MAX(s.order_date) as last_order_date
FROM customers c
LEFT JOIN sales_data s ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.name, c.email
HAVING SUM(s.amount) > 0
ORDER BY total_spent DESC;

-- Verify
SELECT * FROM top_customers LIMIT 10;
