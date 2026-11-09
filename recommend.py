import pandas as pd

# 1. Load the attractions dataset
CSV_FILE = "travel_planner_attractions_dataset.csv"

try: 
    attractions = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Could not find '{CSV_FILE}'.")
    print("Keep the CSV file in the same folder as recommend.py.")
    raise SystemExit(1)

# 2. Ask the user for travel preferences
destination = input("Enter destination (Bengaluru or Mysuru): ").strip()
interests_input = input(
    "Enter your interests separated by commas (example: nature, history): "
).strip()

# Split the interests into individual words/phrases
interests = [
    item.strip().lower()
    for item in interests_input.split(",")
    if item.strip()
]

budget_input = input(
    "Enter your budget for attraction entry fees per person in INR "
    "(or press Enter to skip): "
).strip()

budget = None
if budget_input:
    try:
        budget = float(budget_input)
        if budget < 0:
            print("Budget cannot be negative.")
            raise SystemExit(1)
    except ValueError:
        print("Please enter a valid number for the budget.")
        raise SystemExit(1)

# 3. Find attractions in the selected destination
matches = attractions[
    attractions["destination"].str.strip().str.lower()
    == destination.lower()
].copy()

if matches.empty:
    print(f"\nNo attractions found for '{destination}'.")
    print("Try Bengaluru or Mysuru, which are in this starter dataset.")
    raise SystemExit(0)

# 4. Score attractions by matching interests with category, keywords, and tags
def calculate_match_score(row, user_interests):
    searchable_text = " ".join([
        str(row.get("category", "")),
        str(row.get("keywords", "")),
        str(row.get("tags", "")),
        str(row.get("attraction_name", "")),
    ]).lower()

    score = 0
    for interest in user_interests:
        if interest in searchable_text:
            score += 1
    return score

matches["match_score"] = matches.apply(
    lambda row: calculate_match_score(row, interests), axis=1
)

# 5. If the user supplied a budget, filter by estimated entry fee per person.
# This does not include transport, food, accommodation, or shopping.
if budget is not None:
    matches = matches[
        matches["estimated_entry_fee_inr_per_person"] <= budget
    ]

# 6. Show the best matches first
matches = matches.sort_values(
    by=["match_score", "estimated_visit_hours"],
    ascending=[False, True]
)

recommended = matches[matches["match_score"] > 0].head(10)

print("\n========== TRAVEL RECOMMENDATIONS ==========")
print(f"Destination: {destination.title()}")
print(f"Interests: {', '.join(interests) if interests else 'Not specified'}")

if budget is not None:
    print(f"Entry-fee budget per person: ₹{budget:.0f}")

if recommended.empty:
    print("\nNo attractions matched those interests and budget.")
    print("Try broader interests such as nature, history, art, or wildlife.")
else:
    total_estimated_entry_fees = 0

    for number, (_, place) in enumerate(recommended.iterrows(), start=1):
        fee = float(place["estimated_entry_fee_inr_per_person"])
        hours = place["estimated_visit_hours"]
        total_estimated_entry_fees += fee

        print(f"\n{number}. {place['attraction_name']}")
        print(f"   Category: {place['category']}")
        print(f"   Estimated entry fee per person: ₹{fee:.0f}")
        print(f"   Estimated visit time: {hours} hours")
        print(f"   Matched interests score: {int(place['match_score'])}")

    print("\n--------------------------------------------")
    print(
        "Sum of listed entry fees for these recommendations: "
        f"₹{total_estimated_entry_fees:.0f} per person"
    )
    print("Costs are starter estimates; verify current fees before travel.")
    print("Transport, food, accommodation, and shopping are not included.")
