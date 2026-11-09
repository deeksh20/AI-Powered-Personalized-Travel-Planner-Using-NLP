import pandas as pd

CSV_FILE = "travel_planner_attractions_dataset.csv"

# 1. Load the dataset
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"ERROR: Could not find {CSV_FILE}")
    print("Put the CSV file in the same folder as this Python file.")
    raise SystemExit(1)

# 2. Collect travel preferences
destination = input("Enter destination (Bengaluru or Mysuru): ").strip()
interest_input = input("Enter interests separated by commas (e.g. nature, history): ").strip()
interests = [x.strip().lower() for x in interest_input.split(",") if x.strip()]

try:
    days = int(input("How many days is your trip? "))
    if days < 1:
        print("Trip duration must be at least 1 day.")
        raise SystemExit(1)
except ValueError:
    print("Please enter the number of days as a whole number.")
    raise SystemExit(1)

budget_input = input("Total entry-fee budget per person in INR (or press Enter to skip): ").strip()
budget = None
if budget_input:
    try:
        budget = float(budget_input)
        if budget < 0:
            print("Budget cannot be negative.")
            raise SystemExit(1)
    except ValueError:
        print("Please enter a valid budget number.")
        raise SystemExit(1)

# 3. Filter by destination
places = df[df["destination"].astype(str).str.strip().str.lower() == destination.lower()].copy()

if places.empty:
    print("\nDestination not found. This starter dataset supports Bengaluru and Mysuru.")
    raise SystemExit(0)

# 4. Match interests against attraction name, category, keywords and tags
def calculate_score(row):
    text = " ".join([
        str(row.get("attraction_name", "")),
        str(row.get("category", "")),
        str(row.get("keywords", "")),
        str(row.get("tags", ""))
    ]).lower()
    return sum(1 for interest in interests if interest in text)

places["match_score"] = places.apply(calculate_score, axis=1)
places = places[places["match_score"] > 0].copy()

if places.empty:
    print("\nNo attractions matched those interests. Try nature, history, art, wildlife or architecture.")
    raise SystemExit(0)

# 5. Rank strongest matches first
places = places.sort_values(
    by=["match_score", "estimated_visit_hours"],
    ascending=[False, True]
)

# 6. Build a simple itinerary.
# For this first version, allow about 6 hours of attraction visits each day.
DAILY_VISIT_HOUR_LIMIT = 6.0
itinerary = {day: [] for day in range(1, days + 1)}
day_hours = {day: 0.0 for day in range(1, days + 1)}
total_fees = 0.0

for _, place in places.iterrows():
    fee = float(place["estimated_entry_fee_inr_per_person"])
    visit_hours = float(place["estimated_visit_hours"])

    # Do not exceed the user's total entry-fee budget.
    if budget is not None and total_fees + fee > budget:
        continue

    # Put this attraction on the first day with enough remaining time.
    assigned = False
    for day in range(1, days + 1):
        if day_hours[day] + visit_hours <= DAILY_VISIT_HOUR_LIMIT:
            itinerary[day].append(place)
            day_hours[day] += visit_hours
            total_fees += fee
            assigned = True
            break

    # If no day has enough time, skip this attraction in this simple version.
    if not assigned:
        continue

# 7. Display the itinerary
print("\n========== YOUR PERSONALIZED ITINERARY ==========")
print(f"Destination: {destination.title()}")
print(f"Trip duration: {days} day(s)")
print("Interests:", ", ".join(interests) if interests else "Not specified")
print("Daily attraction visit-time limit:", DAILY_VISIT_HOUR_LIMIT, "hours")

for day in range(1, days + 1):
    print(f"\n--- DAY {day} ---")
    if not itinerary[day]:
        print("No attractions assigned. Try a broader interest, more days, or a higher budget.")
        continue

    for number, place in enumerate(itinerary[day], start=1):
        print(f"{number}. {place['attraction_name']} ({place['category']})")
        print(f"   Estimated entry fee: ₹{float(place['estimated_entry_fee_inr_per_person']):.0f} per person")
        print(f"   Estimated visit time: {float(place['estimated_visit_hours']):.1f} hour(s)")

    print(f"Estimated attraction visit time: {day_hours[day]:.1f} hours")

print("\n========== COST SUMMARY ==========")
print(f"Estimated total entry fees per person: ₹{total_fees:.0f}")
if budget is not None:
    print(f"Your entry-fee budget: ₹{budget:.0f}")
    print(f"Estimated budget remaining: ₹{budget - total_fees:.0f}")

print("\nImportant:")
print("- Costs and visit durations are starter estimates; verify current prices.")
print("- Budget covers attraction entry fees only, not travel, food, accommodation or shopping.")
print("- This is a basic prototype. It does not yet optimize routes or check opening hours.")
