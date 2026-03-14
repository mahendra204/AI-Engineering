# Module 3.2: Spark SQL Fundamentals

## 🗄️ Spark SQL Overview

Spark SQL provides unified interface for:
- SQL queries
- DataFrames
- External data sources
- Optimization via Catalyst

---

## 📝 Creating Tables

### Temporary Views

```sql
-- CREATE TEMPORARY VIEW (session scope)
CREATE TEMPORARY VIEW temp_orders AS
SELECT * FROM orders WHERE amount > 100;

-- Use
SELECT * FROM temp_orders;

-- Drop
DROP VIEW temp_orders;
```

### Global Temporary Views

```sql
-- Global scope (across sessions in same cluster)
CREATE GLOBAL TEMPORARY VIEW global_orders AS
SELECT * FROM orders;

-- Access with namespace
SELECT * FROM global_temp.global_orders;
```

### Persistent Tables

```sql
-- Managed table (stored in warehouse)
CREATE TABLE customers (
    customer_id INT,
    name STRING,
    email STRING COMMENT 'Customer email',
    created_date DATE
)
USING DELTA
PARTITIONED BY (created_date);

-- External table (stored in DBFS/cloud)
CREATE TABLE customers_external (
    customer_id INT,
    name STRING
)
USING DELTA
LOCATION '/mnt/data/customers/';

-- Table from query
CREATE TABLE high_value_customers AS
SELECT * FROM customers
WHERE lifetime_value > 10000;
```

---

## 📊 SQL Queries

### SELECT & WHERE

```sql
-- Basic select
SELECT customer_id, name, email FROM customers;

-- With WHERE
SELECT * FROM customers WHERE created_date >= '2023-01-01';

-- Multiple conditions
SELECT * FROM orders
WHERE amount > 100 AND status = 'completed' OR priority = 'high';

-- IN operator
SELECT * FROM customers WHERE region IN ('US', 'CA', 'MX');

-- LIKE pattern
SELECT * FROM customers WHERE email LIKE '%@gmail.com';
```

### DISTINCT & SORTING

```sql
-- Remove duplicates
SELECT DISTINCT region FROM customers;

-- Sort
SELECT * FROM orders ORDER BY amount DESC, date ASC;

-- LIMIT
SELECT * FROM customers LIMIT 100;

-- OFFSET (skip rows)
SELECT * FROM customers ORDER BY created_date DESC LIMIT 10 OFFSET 20;
```

### GROUP BY & HAVING

```sql
-- Group by
SELECT region, COUNT(*) as cnt, SUM(amount) as total
FROM orders
GROUP BY region;

-- HAVING (filter aggregates)
SELECT region, SUM(amount) as total
FROM orders
GROUP BY region
HAVING SUM(amount) > 100000;

-- Multiple grouping levels
SELECT region, product, COUNT(*) as orders
FROM sales
GROUP BY region, product
ORDER BY region, orders DESC;
```

### JOINS

```sql
-- INNER JOIN
SELECT o.order_id, o.amount, c.name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id;

-- LEFT JOIN
SELECT o.order_id, c.name
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id;

-- FULL OUTER JOIN
SELECT *
FROM orders o
FULL OUTER JOIN customers c ON o.customer_id = c.customer_id;

-- CROSS JOIN (all combinations)
SELECT a.region, b.product
FROM regions a
CROSS JOIN products b;

-- Self-join
SELECT e1.name as employee, e2.name as manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.employee_id;
```

### Subqueries & CTEs

```sql
-- Subquery
SELECT * FROM customers
WHERE customer_id IN (
    SELECT customer_id FROM orders WHERE amount > 1000
);

-- CTE (Common Table Expression)
WITH high_value_orders AS (
    SELECT customer_id, SUM(amount) as total
    FROM orders
    GROUP BY customer_id
    HAVING SUM(amount) > 10000
)
SELECT c.name, hvo.total
FROM high_value_orders hvo
JOIN customers c ON hvo.customer_id = c.customer_id;

-- Multiple CTEs
WITH
sales_by_region AS (
    SELECT region, SUM(amount) as total FROM sales GROUP BY region
),
top_regions AS (
    SELECT * FROM sales_by_region WHERE total > 100000
)
SELECT * FROM top_regions;
```

### UNION & SET OPERATIONS

```sql
-- UNION (distinct rows)
SELECT name, email FROM customers
UNION
SELECT contact_name, email FROM vendors;

-- UNION ALL (all rows)
SELECT * FROM orders_2023
UNION ALL
SELECT * FROM orders_2024;

-- INTERSECT (common rows)
SELECT id FROM current_customers
INTERSECT
SELECT id FROM previous_customers;

-- EXCEPT (rows in first not in second)
SELECT id FROM all_customers
EXCEPT
SELECT id FROM inactive_customers;
```

---

## 🪟 Window Functions

Most powerful SQL feature for complex analytics!

```sql
-- ROW_NUMBER (rank with ties)
SELECT
    employee_id,
    salary,
    department_id,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as rank
FROM employees;

-- RANK (same rank for ties)
SELECT
    employee_id,
    salary,
    RANK() OVER (ORDER BY salary DESC) as salary_rank
FROM employees;

-- Running total
SELECT
    date,
    amount,
    SUM(amount) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total
FROM sales
ORDER BY date;

-- Moving average
SELECT
    date,
    close_price,
    AVG(close_price) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as moving_avg_30
FROM stock_prices;

-- LAG and LEAD (previous/next row)
SELECT
    date,
    close_price,
    LAG(close_price) OVER (ORDER BY date) as prev_price,
    LEAD(close_price) OVER (ORDER BY date) as next_price
FROM stock_prices;

-- Percentile
SELECT
    employee_id,
    salary,
    PERCENT_RANK() OVER (ORDER BY salary) as salary_percentile
FROM employees;
```

---

## 📅 Date & Time Functions

```sql
-- Current date/time
SELECT CURRENT_DATE, CURRENT_TIMESTAMP;

-- Parse
SELECT TO_DATE('2023-01-15', 'yyyy-MM-dd');

-- Format
SELECT DATE_FORMAT(order_date, 'yyyy-MM-dd HH:mm:ss');

-- Extract
SELECT YEAR(order_date), MONTH(order_date), DAY(order_date);

-- Calculate
SELECT DATE_ADD(order_date, 30) as delivery_date;
SELECT DATEDIFF(CURRENT_DATE, order_date) as days_ago;
SELECT DATE_TRUNC('month', order_date) as month_start;
```

---

## 🔀 String Functions

```sql
-- Case
SELECT UPPER(name), LOWER(email), INITCAP(city);

-- Substring
SELECT SUBSTRING(phone, 1, 3) as area_code;

-- Replace
SELECT REPLACE(email, '@old.com', '@new.com');

-- Trim
SELECT TRIM(name), LTRIM(name), RTRIM(name);

-- Concat
SELECT CONCAT(first_name, ' ', last_name) as full_name;

-- Split
SELECT SPLIT(email, '@')[0] as username;

-- LIKE
SELECT * FROM customers WHERE email LIKE '%@gmail.com';
```

---

## 📊 Aggregate Functions

```sql
-- Basic aggregates
SELECT
    COUNT(*) as total_rows,
    COUNT(DISTINCT customer_id) as unique_customers,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    MIN(amount) as min_amount,
    MAX(amount) as max_amount,
    STDDEV(amount) as std_deviation
FROM orders;

-- Conditional aggregates
SELECT
    SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) as completed_amount,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_count
FROM orders;
```

---

## 🆕 Recent SQL Features (2024-2025)

### Lateral Columns Alias (2024)
```sql
SELECT
    customer_id,
    amount * 1.1 as amount_with_tax,      -- First definition
    amount_with_tax * 0.08 as tax_amount  -- Can reference above!
FROM orders;
```

### STRUCT & MAP (Nested Data)
```sql
SELECT
    customer_id,
    STRUCT(
        first_name,
        last_name,
        email
    ) as customer_info,
    MAP(
        'region', region,
        'status', status
    ) as attributes
FROM customers;
```

### JSON Functions
```sql
SELECT
    customer_id,
    GET_JSON_OBJECT(json_data, '$.name') as name,
    TO_JSON(STRUCT(id, name, email)) as json_output
FROM customers;
```

---

## 🎯 Best Practices

✅ Use CTEs for readability  
✅ Use window functions instead of self-joins  
✅ Index frequently filtered columns  
✅ Use EXPLAIN to understand query plans  
✅ Avoid SELECT * in production queries  
✅ Use appropriate data types  
✅ Partition large tables  

---

## 🧪 Hands-on Lab

```sql
-- Lab: Complex SQL Analysis

-- Create sample data
CREATE TEMPORARY VIEW sales_data AS
SELECT * FROM (
    VALUES
        (1, 'Alice', 'US', 150.50, '2023-01-15'),
        (2, 'Bob', 'CA', 200.75, '2023-01-16'),
        (1, 'Alice', 'US', 175.25, '2023-01-17'),
        (3, 'Charlie', 'MX', 125.00, '2023-01-18'),
        (2, 'Bob', 'CA', 300.00, '2023-01-19')
) AS data(customer_id, name, region, amount, date);

-- Analysis 1: Total by region
SELECT
    region,
    COUNT(*) as num_orders,
    SUM(amount) as total_sales,
    AVG(amount) as avg_sale
FROM sales_data
GROUP BY region
ORDER BY total_sales DESC;

-- Analysis 2: With window functions
SELECT
    name,
    amount,
    SUM(amount) OVER (PARTITION BY name ORDER BY date) as customer_total,
    ROW_NUMBER() OVER (PARTITION BY name ORDER BY date) as order_number
FROM sales_data
ORDER BY name, date;

-- Analysis 3: Customer ranking
WITH customer_metrics AS (
    SELECT
        customer_id,
        name,
        COUNT(*) as num_orders,
        SUM(amount) as total_amount
    FROM sales_data
    GROUP BY customer_id, name
)
SELECT
    *,
    RANK() OVER (ORDER BY total_amount DESC) as customer_rank
FROM customer_metrics;
```

---

**Duration:** 2 hours | **Difficulty:** Intermediate-Advanced | **Last Updated:** 2025

