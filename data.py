import random
import json

# Sample clauses pool for demonstration
CLAUSES = [
    {
        "clause_number": "5.2",
        "text": "Expenses incurred for cataract surgery shall be payable after continuous coverage of 12 months from policy inception, subject to a sub-limit of INR 25,000 per eye per policy year."
    },
    {
        "clause_number": "6.4",
        "text": "Joint replacement surgeries are covered only after 24 months of continuous coverage except where arising from accidents."
    },
    {
        "clause_number": "3.1",
        "text": "Hospitalization expenses for any illness not specifically excluded are covered after a waiting period of 30 days from first policy inception."
    },
    {
        "clause_number": "7.3",
        "text": "Bariatric surgery is covered after 48 months of continuous coverage, only if medically necessary and not for cosmetic purposes."
    },
    {
        "clause_number": "8.1",
        "text": "Maternity expenses are covered after a 24-month waiting period from policy inception."
    },
    {
        "clause_number": "9.5",
        "text": "Outpatient dental treatments are excluded from coverage except where required due to accidental injury."
    },
    {
        "clause_number": "10.2",
        "text": "Pre-existing diseases are covered after 36 months of continuous coverage."
    },
    {
        "clause_number": "4.3",
        "text": "Accidental injury-related hospitalizations are covered from day 1 without any waiting period."
    },
    {
        "clause_number": "12.1",
        "text": "Organ donor expenses are covered up to INR 100,000 per policy year."
    },
    {
        "clause_number": "14.2",
        "text": "Inpatient psychiatric treatments are covered after 24 months of continuous coverage."
    }
]

# Sample procedures, conditions, cities, and age ranges
PROCEDURES = [
    "cataract surgery", "knee replacement", "bariatric surgery",
    "maternity expenses", "dental treatment", "psychiatric treatment",
    "organ donor surgery", "hospitalization for dengue", "accidental injury"
]
CITIES = [
    "Mumbai", "Pune", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Ahmedabad", "Nagpur", "Lucknow"
]
AGE_RANGE = list(range(18, 75))

# Decision templates
DECISION_TEMPLATES = [
    {
        "decision": "Approved",
        "amount": "As per policy limits",
        "justification": "The procedure is covered as per the relevant policy clause. The policyholder meets all the specified conditions."
    },
    {
        "decision": "Rejected",
        "amount": "0",
        "justification": "The procedure is not covered due to waiting period, exclusion, or insufficient policy duration according to the relevant clause."
    }
]

def random_policy_duration():
    # 1 to 60 months (5 years)
    return random.choice([3, 6, 12, 24, 36, 48, 60])

def random_gender():
    return random.choice(["male", "female"])

def make_query(age, gender, procedure, city, months):
    short_gender = "M" if gender == "male" else "F"
    return f"{age}-year-old {short_gender}, {procedure}, {city}, {months}-month policy"

def make_answer_and_clause(procedure, months):
    # Map procedure to clause and logic
    mapping = {
        "cataract surgery": (0, "Approved" if months >= 12 else "Rejected"),
        "knee replacement": (1, "Approved" if months >= 24 else "Rejected"),
        "hospitalization for dengue": (2, "Approved" if months >= 1 else "Rejected"),
        "bariatric surgery": (3, "Approved" if months >= 48 else "Rejected"),
        "maternity expenses": (4, "Approved" if months >= 24 else "Rejected"),
        "dental treatment": (5, "Rejected"),
        "organ donor surgery": (8, "Approved" if months >= 12 else "Rejected"),
        "psychiatric treatment": (9, "Approved" if months >= 24 else "Rejected"),
        "accidental injury": (7, "Approved"),
    }
    clause_idx, decision = mapping.get(procedure, (None, "Rejected"))
    clause = CLAUSES[clause_idx] if clause_idx is not None else {"clause_number": "N/A", "text": "No relevant clause found."}
    if decision == "Approved":
        amount = "As per policy limits"
        justification = f"{procedure.title()} is covered as per clause {clause['clause_number']}. The policyholder meets the waiting period and eligibility criteria."
    else:
        amount = "0"
        justification = f"{procedure.title()} is not covered as per clause {clause['clause_number']} due to waiting period, exclusion, or insufficient policy duration."
    return {
        "decision": decision,
        "amount": amount,
        "justification": justification
    }, [clause]

def generate_dataset(n=10000):
    dataset = []
    for _ in range(n):
        age = random.choice(AGE_RANGE)
        gender = random_gender()
        procedure = random.choice(PROCEDURES)
        city = random.choice(CITIES)
        months = random_policy_duration()
        query = make_query(age, gender, procedure, city, months)
        answer, clauses = make_answer_and_clause(procedure, months)
        row = {
            "query": query,
            "answer": answer,
            "clauses": clauses
        }
        dataset.append(row)
    return dataset

if __name__ == "__main__":
    N = 10000 # or any number you want
    data = generate_dataset(N)
    with open("insurance_qa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Generated {N} insurance queries with context and clauses in insurance_qa_dataset.json")