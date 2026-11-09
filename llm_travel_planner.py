
import json
import pandas as pd
import ollama

# 1. Load our real attractions dataset
df = pd.read_csv("travel_planner_attractions_dataset.csv")

# 2. Get the user's travel preferences
destination = input("Destination (Bengaluru or Mysuru): ").strip()
interests = input("Interests (e.g. nature, history): ").strip()
days = int(input("Number of days: "))
budget = float(input("Entry-fee budget per person (₹): "))

# 3. Retrieve attractions from the CSV
places = df[
    df["destination"].str.lower().str.strip()
    == destination.lower()
].copy()

if places.empty:
    print("Destination not found in the dataset.")
    raise SystemExit

# Filter to attractions that individually fit the budget
places = places[
    places["estimated_entry_fee_inr_per_person"] <= budget
].copy()

if places.empty:
    print("No attractions fit this entry-fee budget.")
    raise SystemExit

# Give the LLM only verified rows from our dataset
columns = [
    "attraction_name",
    "category",
    "keywords",
    "estimated_entry_fee_inr_per_person",
    "estimated_visit_hours",
]
catalog = places[columns].to_dict(orient="records")

# 4. Ask the LLM to plan using only the supplied catalog
prompt = f"""
You are a personalized travel-planning assistant.

Create a {days}-day itinerary for {destination}.
User interests: {interests}
Total entry-fee budget per person: ₹{budget:.0f}

Use ONLY attractions in this catalog:
{json.dumps(catalog, ensure_ascii=False)}

Rules:
- Do not invent attraction names or prices.
- Use the exact attraction names and costs from the catalog.
- Keep the total entry fees within the budget.
- Arrange visits across the requested number of days.
- Aim for no more than 6 hours of attraction visits per day.
- Select places relevant to the user's interests.
- Return valid JSON only, with this structure:
{{
  "itinerary": [
    {{
      "day": 1,
      "attractions": [
        {{
          "attraction_name": "Exact catalog name",
          "reason": "Why it matches the user's interests",
          "estimated_cost": 0,
          "visit_duration_hours": 1.0
        }}
      ]
    }}
  ]
}}
"""

# 5. Generate the itinerary with Ollama
response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}],
    format="json",
)

# 6. Parse and display the response
try:
    result = json.loads(response["message"]["content"])
    print("\n===== LLM-GENERATED ITINERARY =====")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except (json.JSONDecodeError, KeyError, TypeError):
    print("The model returned an unexpected response.")
    print(response["message"]["content"])
