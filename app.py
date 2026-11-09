import json
import pandas as pd
import streamlit as st
import ollama

DATASET_FILE = "travel_planner_attractions_dataset.csv"
MODEL_NAME = "llama3.2"
DAILY_HOUR_LIMIT = 6.0

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="🧭",
    layout="wide"
)

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATASET_FILE)
    required = {
        "destination", "attraction_name", "category", "keywords", "tags",
        "estimated_entry_fee_inr_per_person", "estimated_visit_hours"
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(sorted(missing)))
    data["estimated_entry_fee_inr_per_person"] = pd.to_numeric(
        data["estimated_entry_fee_inr_per_person"], errors="coerce"
    )
    data["estimated_visit_hours"] = pd.to_numeric(
        data["estimated_visit_hours"], errors="coerce"
    )
    data = data.dropna(
        subset=["destination", "attraction_name",
                "estimated_entry_fee_inr_per_person", "estimated_visit_hours"]
    )
    return data

st.title("🧭 AI-Powered Personalized Travel Planner")
st.caption("Personalized itineraries powered by your attractions dataset and a locally running Llama 3.2 model.")

try:
    df = load_dataset()
except FileNotFoundError:
    st.error(f"Dataset not found: {DATASET_FILE}. Put the CSV beside app.py.")
    st.stop()
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

with st.sidebar:
    st.header("Plan your trip")
    destinations = sorted(df["destination"].dropna().astype(str).unique().tolist())
    destination = st.selectbox("Destination", destinations)
    interest_text = st.text_input("Interests", placeholder="nature, history, photography")
    days = st.number_input("Trip duration (days)", min_value=1, max_value=14, value=2, step=1)
    budget = st.number_input(
        "Entry-fee budget per person (₹)",
        min_value=0,
        max_value=1000000,
        value=500,
        step=100
    )
    generate = st.button("Generate my itinerary", type="primary", use_container_width=True)

st.info(
    "Budget currently covers estimated attraction entry fees only. "
    "Transport, food, accommodation, and shopping are not included."
)

matching_destination = df[
    df["destination"].astype(str).str.strip().str.casefold()
    == destination.casefold()
].copy()

st.subheader(f"Explore {destination}")
st.write(f"**{len(matching_destination)} attractions** available in the current dataset.")

with st.expander("Preview attractions in the dataset"):
    preview_columns = [
        "attraction_name", "category",
        "estimated_entry_fee_inr_per_person", "estimated_visit_hours"
    ]
    st.dataframe(
        matching_destination[preview_columns].rename(columns={
            "attraction_name": "Attraction",
            "category": "Category",
            "estimated_entry_fee_inr_per_person": "Estimated entry fee (₹)",
            "estimated_visit_hours": "Visit time (hours)"
        }),
        use_container_width=True,
        hide_index=True
    )

if generate:
    interests = [x.strip().lower() for x in interest_text.split(",") if x.strip()]

    candidates = matching_destination[
        (matching_destination["estimated_entry_fee_inr_per_person"] >= 0)
        & (matching_destination["estimated_visit_hours"] > 0)
        & (matching_destination["estimated_entry_fee_inr_per_person"] <= float(budget))
    ].copy()

    if candidates.empty:
        st.warning("No attractions individually fit this entry-fee budget. Increase the budget or try another destination.")
        st.stop()

    catalog_columns = [
        "attraction_name", "category", "keywords", "tags",
        "estimated_entry_fee_inr_per_person", "estimated_visit_hours"
    ]
    catalog = candidates[catalog_columns].to_dict(orient="records")

    prompt = f"""
You are a careful travel itinerary planner.
Create a {int(days)}-day itinerary for {destination}.
User interests: {", ".join(interests) if interests else "general sightseeing"}.
Total entry-fee budget per person: INR {float(budget):.0f}.

Use ONLY attractions from this catalog:
{json.dumps(catalog, ensure_ascii=False)}

Return valid JSON only in this structure:
{{"itinerary":[{{"day":1,"attractions":[{{"attraction_name":"exact catalog name",
"reason":"brief explanation of interest match"}}]}}]}}
Rules:
- Use exact attraction names from the catalog.
- Do not invent attractions, prices, or durations.
- Do not repeat attractions.
- Use day numbers from 1 to {int(days)}.
- Aim for at most {DAILY_HOUR_LIMIT} hours of visits per day.
- Keep total entry fees within the budget.
"""

    with st.spinner("Asking Llama 3.2 to create your itinerary..."):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json"
            )
            result = json.loads(response["message"]["content"])
        except Exception as error:
            st.error(
                "Could not contact Ollama or parse the model response. "
                "Make sure Ollama is running and `ollama run llama3.2` has completed at least once."
            )
            st.code(str(error))
            st.stop()

    # Validate every selected attraction against the dataset.
    catalog_by_name = {
        str(row["attraction_name"]).strip().casefold(): row
        for _, row in candidates.iterrows()
    }
    validated_days = {day: [] for day in range(1, int(days) + 1)}
    used_names = set()
    total_cost = 0.0
    validation_notes = []

    model_days = result.get("itinerary", [])
    if not isinstance(model_days, list):
        model_days = []
        validation_notes.append("Model returned an invalid itinerary structure.")

    for day_item in model_days:
        if not isinstance(day_item, dict):
            validation_notes.append("Skipped a malformed day entry.")
            continue
        try:
            day_number = int(day_item.get("day"))
        except (TypeError, ValueError):
            validation_notes.append("Skipped a day with an invalid day number.")
            continue
        if day_number not in validated_days:
            validation_notes.append(f"Skipped invalid day number: {day_number}.")
            continue

        items = day_item.get("attractions", [])
        if not isinstance(items, list):
            validation_notes.append(f"Skipped malformed attraction list on Day {day_number}.")
            continue

        for item in items:
            if not isinstance(item, dict):
                validation_notes.append("Skipped a malformed attraction entry.")
                continue
            name = str(item.get("attraction_name", "")).strip()
            key = name.casefold()

            if key not in catalog_by_name:
                validation_notes.append(f"Rejected unknown attraction: {name or '(blank name)'}")
                continue
            if key in used_names:
                validation_notes.append(f"Rejected duplicate attraction: {name}")
                continue

            row = catalog_by_name[key]
            fee = float(row["estimated_entry_fee_inr_per_person"])
            hours = float(row["estimated_visit_hours"])
            day_hours = sum(place["hours"] for place in validated_days[day_number])

            if day_hours + hours > DAILY_HOUR_LIMIT:
                validation_notes.append(f"Skipped {name}: daily visit-time limit exceeded.")
                continue
            if total_cost + fee > float(budget):
                validation_notes.append(f"Skipped {name}: total entry-fee budget exceeded.")
                continue

            reason = str(item.get("reason", "")).strip()
            if not reason:
                reason = "Selected from the dataset for your trip."
            validated_days[day_number].append({
                "name": str(row["attraction_name"]),
                "category": str(row["category"]),
                "fee": fee,
                "hours": hours,
                "reason": reason
            })
            used_names.add(key)
            total_cost += fee

    st.subheader("Your personalized itinerary")
    if not used_names:
        st.warning("No attractions passed validation. Try broader interests or a larger budget.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Validated attractions", len(used_names))
        col2.metric("Estimated entry fees", f"₹{total_cost:.0f}")
        col3.metric("Budget remaining", f"₹{float(budget) - total_cost:.0f}")

        for day_number in range(1, int(days) + 1):
            entries = validated_days[day_number]
            with st.expander(f"Day {day_number} — {len(entries)} attraction(s)", expanded=True):
                if not entries:
                    st.write("No attractions assigned to this day.")
                    continue
                day_cost = sum(item["fee"] for item in entries)
                day_hours = sum(item["hours"] for item in entries)
                for item in entries:
                    st.markdown(f"**{item['name']}** · {item['category']}")
                    st.write(item["reason"])
                    st.caption(
                        f"Estimated entry fee: ₹{item['fee']:.0f} per person · "
                        f"Estimated visit time: {item['hours']:.1f} hours"
                    )
                st.write(f"**Day estimate:** ₹{day_cost:.0f} · {day_hours:.1f} visit hours")

        st.success("Accepted attraction names, entry fees, and durations were checked against your CSV dataset.")
        st.caption("This prototype does not optimize travel routes or verify live prices/opening hours.")

    if validation_notes:
        with st.expander("Validation notes"):
            for note in validation_notes:
                st.write("• " + note)

    # Download the displayed itinerary as JSON.
    export_data = {
        "destination": destination,
        "interests": interests,
        "days": int(days),
        "entry_fee_budget_per_person_inr": float(budget),
        "estimated_total_entry_fees_per_person_inr": total_cost,
        "itinerary": [
            {
                "day": day_number,
                "attractions": [
                    {
                        "attraction_name": item["name"],
                        "category": item["category"],
                        "reason": item["reason"],
                        "estimated_cost_inr": item["fee"],
                        "visit_duration_hours": item["hours"]
                    }
                    for item in validated_days[day_number]
                ]
            }
            for day_number in range(1, int(days) + 1)
        ]
    }
    st.download_button(
        "Download itinerary as JSON",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name="travel_itinerary.json",
        mime="application/json"
    )

st.divider()
st.caption("Academic prototype • Attraction prices are starter estimates and should be verified before real travel.")
