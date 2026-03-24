"""
generate_employees.py
Generates synthetic employee/HR data and saves to employees.csv
"""

import csv
import random
import uuid
from datetime import datetime, timedelta

# --- Config ---
NUM_EMPLOYEES = 300
OUTPUT_FILE = "employees.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# --- Data pools ---
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "David", "Susan", "Priya", "Arjun", "Sneha", "Rahul", "Ananya", "Vikram",
    "Aisha", "Mohammed", "Fatima", "Omar", "Yuki", "Kenji", "Elena", "Lucas",
    "Isabella", "Mateo", "Sofia", "Wei", "Ming", "Amara", "Chioma", "Kwame"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Patel", "Sharma", "Singh", "Kumar", "Shah", "Mehta", "Reddy", "Khan",
    "Ali", "Ahmed", "Yamamoto", "Tanaka", "Rodriguez", "Martinez", "Wilson",
    "Anderson", "Taylor", "Thomas", "Jackson", "White", "Harris", "Martin", "Chen"
]

DEPARTMENTS = {
    "Engineering": ["Software Engineer", "Senior Software Engineer", "Staff Engineer", "Engineering Manager", "VP Engineering"],
    "Product": ["Product Manager", "Senior PM", "Director of Product", "VP Product"],
    "Sales": ["Sales Rep", "Account Executive", "Sales Manager", "VP Sales", "Chief Revenue Officer"],
    "Marketing": ["Marketing Analyst", "Growth Manager", "Marketing Director", "CMO"],
    "Finance": ["Financial Analyst", "Senior Analyst", "Finance Manager", "CFO"],
    "HR": ["HR Coordinator", "HR Business Partner", "HR Manager", "CHRO"],
    "Operations": ["Operations Analyst", "Operations Manager", "COO"],
    "Customer Success": ["CS Rep", "CS Manager", "Director of CS", "VP Customer Success"],
    "Data": ["Data Analyst", "Data Scientist", "Senior Data Scientist", "Head of Data"],
    "Design": ["UX Designer", "Senior Designer", "Design Lead", "Head of Design"],
}

SENIORITY_SALARY = {
    "Junior": (35000, 60000),
    "Mid": (60000, 95000),
    "Senior": (90000, 140000),
    "Lead/Manager": (120000, 180000),
    "Director/VP": (160000, 250000),
    "C-Level": (220000, 500000),
}

LOCATIONS = [
    "New York, USA", "San Francisco, USA", "Austin, USA", "Seattle, USA",
    "London, UK", "Berlin, Germany", "Mumbai, India", "Bengaluru, India",
    "Toronto, Canada", "Sydney, Australia", "Singapore", "Remote"
]

EMPLOYMENT_TYPES = ["Full-Time", "Full-Time", "Full-Time", "Part-Time", "Contractor"]
GENDERS = ["Male", "Female", "Non-Binary", "Prefer not to say"]
GENDER_WEIGHTS = [0.48, 0.44, 0.04, 0.04]

EDUCATION = ["High School", "Associate's", "Bachelor's", "Master's", "PhD", "Bootcamp/Self-taught"]
EDU_WEIGHTS = [0.05, 0.05, 0.45, 0.30, 0.10, 0.05]


def random_date(start_year=2010, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).date()


def get_seniority(title):
    t = title.lower()
    if any(x in t for x in ["chief", "cto", "cfo", "coo", "cmo", "cro", "chro"]):
        return "C-Level"
    if any(x in t for x in ["director", "vp", "vice president"]):
        return "Director/VP"
    if any(x in t for x in ["manager", "lead", "head"]):
        return "Lead/Manager"
    if any(x in t for x in ["senior", "staff", "sr."]):
        return "Senior"
    if any(x in t for x in ["junior", "jr.", "coordinator", "rep", "analyst"]):
        return "Junior"
    return "Mid"


def generate_employee():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    dept = random.choice(list(DEPARTMENTS.keys()))
    title = random.choice(DEPARTMENTS[dept])
    seniority = get_seniority(title)
    salary_low, salary_high = SENIORITY_SALARY[seniority]
    salary = round(random.uniform(salary_low, salary_high), -2)

    hire_date = random_date()
    is_active = random.choices([True, False], weights=[0.88, 0.12])[0]
    termination_date = random_date(int(str(hire_date)[:4]), 2024) if not is_active else None

    return {
        "employee_id": str(uuid.uuid4()),
        "first_name": first,
        "last_name": last,
        "email": f"{first.lower()}.{last.lower()}@company.com",
        "gender": random.choices(GENDERS, weights=GENDER_WEIGHTS)[0],
        "date_of_birth": random_date(1965, 2000),
        "department": dept,
        "job_title": title,
        "seniority_level": seniority,
        "employment_type": random.choices(EMPLOYMENT_TYPES)[0],
        "location": random.choice(LOCATIONS),
        "hire_date": hire_date,
        "termination_date": termination_date if termination_date else "",
        "is_active": is_active,
        "annual_salary_usd": int(salary),
        "bonus_percent": random.choice([0, 5, 10, 15, 20, 25]),
        "performance_score": round(random.uniform(1.0, 5.0), 1),
        "education_level": random.choices(EDUCATION, weights=EDU_WEIGHTS)[0],
        "years_experience": random.randint(0, 25),
        "manager_id": "",  # Will be back-filled below
    }


def main():
    employees = [generate_employee() for _ in range(NUM_EMPLOYEES)]

    # Back-fill manager_id (managers are senior employees)
    senior_ids = [e["employee_id"] for e in employees if e["seniority_level"] in ("Lead/Manager", "Director/VP", "C-Level")]
    for e in employees:
        if e["seniority_level"] not in ("C-Level",) and senior_ids:
            e["manager_id"] = random.choice(senior_ids)

    fields = list(employees[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(employees)

    print(f"✅ Generated {NUM_EMPLOYEES} employees → {OUTPUT_FILE}")
    return employees


if __name__ == "__main__":
    main()
