import json
import pandas as pd
import ollama

CSV_FILE = "travel_planner_attractions_dataset.csv"
MODEL_NAME = "llama3.2"
DAILY_HOUR_LIMIT = 6.0

# 1. Load the source dataset
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"ERROR: {CSV_FILE} was not found. Keep it beside this Python file.")
    raise SystemExit(1)

required = {
    "destination", "attraction_name", "category", "keywords", "tags",
    "estimated_entry_fee_inr_per_person", "estimated_visit_hours"
}
missing = required - set(df.columns)
if missing:
    print("Dataset is missing columns:", ", ".join(sorted(missing)))
    raise SystemExit(1)

# 2. Collect preferences
destination = input("Destination (Bengaluru or Mysuru): ").strip()
interest_input = input("Interests (e.g. nature, history): ").strip()
interests = [x.strip().lower() for x in interest_input.split(",") if x.strip()]

try:
    days = int(input("Number of days: "))
    budget = float(input("Total entry-fee budget per person (INR): "))
    if days < 1 or budget < 0:
        raise ValueError("Days must be at least 1 and budget cannot be negative.")
except ValueError as error:
    print("Invalid input:", error)
    raise SystemExit(1)

# 3. Retrieve only attractions from the selected destination
places = df[df["destination"].astype(str).str.strip().str.casefold()
            == destination.casefold()].copy()
if places.empty:
    print("Destination not found. Try Bengaluru or Mysuru.")
    raise SystemExit(0)

places["estimated_entry_fee_inr_per_person"] = pd.to_numeric(
    places["estimated_entry_fee_inr_per_person"], errors="coerce")
places["estimated_visit_hours"] = pd.to_numeric(
    places["estimated_visit_hours"], errors="coerce")
places = places.dropna(subset=[
    "estimated_entry_fee_inr_per_person", "estimated_visit_hours"])
places = places[(places["estimated_entry_fee_inr_per_person"] >= 0)
                & (places["estimated_visit_hours"] > 0)]
places = places[places["estimated_entry_fee_inr_per_person"] <= budget].copy()
if places.empty:
    print("No attractions fit this budget.")
    raise SystemExit(0)

# 4. Ask the LLM to arrange attractions from the supplied catalog
catalog_cols = [
    "attraction_name", "category", "keywords", "tags",
    "estimated_entry_fee_inr_per_person", "estimated_visit_hours"]
catalog = places[catalog_cols].to_dict(orient="records")

prompt = f"""
Plan a {days}-day itinerary for {destination}.
User interests: {', '.join(interests) if interests else 'general sightseeing'}.
Total entry-fee budget per person: INR {budget:.0f}.
Choose ONLY attractions from this catalog:
{json.dumps(catalog, ensure_ascii=False)}

Return valid JSON only:
{{"itinerary":[{{"day":1,"attractions":[{{"attraction_name":"exact catalog name",
"reason":"short reason matching interests"}}]}}]}}
Do not invent attraction names or prices. Do not repeat attractions.
Use day numbers 1 to {days}. Aim for at most {DAILY_HOUR_LIMIT} visit hours per day.
Keep the total catalog entry fees within the budget.
"""

print("\nGenerating itinerary with Ollama...")
try:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        format="json")
    result = json.loads(response["message"]["content"])
except Exception as error:
    print("Could not generate/parse an Ollama response.")
    print("Check Ollama is running and llama3.2 is available.")
    print("Details:", error)
    raise SystemExit(1)

# 5. Validate LLM output. Dataset values are authoritative for costs and duration.
catalog_by_name = {
    str(row["attraction_name"]).strip().casefold(): row
    for _, row in places.iterrows()
}
validated = {day: [] for day in range(1, days + 1)}
used_names = set()
total_cost = 0.0
messages = []

items_by_day = result.get("itinerary", [])
if not isinstance(items_by_day, list):
    items_by_day = []
    messages.append("Invalid itinerary structure returned by the model.")

for day_item in items_by_day:
    if not isinstance(day_item, dict):
        messages.append("Skipped malformed day entry.")
        continue
    try:
        day = int(day_item.get("day"))
    except (TypeError, ValueError):
        messages.append("Skipped an entry with an invalid day number.")
        continue
    if day not in validated:
        messages.append(f"Skipped invalid day number: {day}.")
        continue

    attraction_items = day_item.get("attractions", [])
    if not isinstance(attraction_items, list):
        messages.append(f"Skipped malformed attractions for Day {day}.")
        continue

    for item in attraction_items:
        if not isinstance(item, dict):
            messages.append("Skipped malformed attraction entry.")
            continue
        name = str(item.get("attraction_name", "")).strip()
        key = name.casefold()
        if key not in catalog_by_name:
            messages.append(f"Rejected attraction not in the dataset: {name or '(blank)'}.")
            continue
        if key in used_names:
            messages.append(f"Rejected duplicate attraction: {name}.")
            continue

        row = catalog_by_name[key]
        fee = float(row["estimated_entry_fee_inr_per_person"])
        hours = float(row["estimated_visit_hours"])
        hours_already = sum(x["hours"] for x in validated[day])
        if hours_already + hours > DAILY_HOUR_LIMIT:
            messages.append(f"Skipped {name}: daily time limit exceeded.")
            continue
        if total_cost + fee > budget:
            messages.append(f"Skipped {name}: total budget would be exceeded.")
            continue

        reason = str(item.get("reason", "")).strip()
        if not reason:
            reason = "Suggested based on the selected preferences."
        validated[day].append({
            "name": str(row["attraction_name"]),
            "category": str(row["category"]),
            "fee": fee,
            "hours": hours,
            "reason": reason
        })
        used_names.add(key)
        total_cost += fee

# 6. Print only validated attractions and dataset costs
print("\n========== VALIDATED TRAVEL ITINERARY ==========")
print("Destination:", destination.title())
print("Interests:", ", ".join(interests) if interests else "General sightseeing")
print(f"Trip duration: {days} day(s)")
print(f"Entry-fee budget per person: INR {budget:.0f}")

for day in range(1, days + 1):
    print(f"\n--- DAY {day} ---")
    entries = validated[day]
    if not entries:
        print("No attractions assigned to this day.")
        continue
    day_cost = 0.0
    day_hours = 0.0
    for number, entry in enumerate(entries, start=1):
        print(f"{number}. {entry['name']} ({entry['category']})")
        print("   Why:", entry["reason"])
        print(f"   Dataset entry-fee estimate: INR {entry['fee']:.0f} per person")
        print(f"   Dataset visit-time estimate: {entry['hours']:.1f} hours")
        day_cost += entry["fee"]
        day_hours += entry["hours"]
    print(f"Day entry fees: INR {day_cost:.0f}")
    print(f"Day visit time: {day_hours:.1f} hours")

print("\n========== VALIDATION SUMMARY ==========")
print("Validated attractions:", len(used_names))
print(f"Estimated total entry fees per person: INR {total_cost:.0f}")
print(f"Budget remaining: INR {budget - total_cost:.0f}")
print("Budget respected:", "YES" if total_cost <= budget else "NO")
print("Costs are dataset estimates, not live prices.")
print("Transport, food, accommodation and shopping are excluded.")

if messages:
    print("\nItems adjusted or rejected during validation:")
    for message in messages:
        print("-", message)
else:
    print("\nAll returned attraction names and schedule constraints passed validation.")
