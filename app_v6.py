import json, math
from datetime import datetime
import pandas as pd
import streamlit as st
import ollama

DATASET_FILE = "travel_planner_attractions_dataset.csv"
MODEL = "llama3.2"
MAX_DAY_HOURS = 8.0
AVG_SPEED_KMH = 25.0
ROAD_FACTOR = 1.3

# Approximate coordinates for demo scheduling; not live navigation data.
COORDS = {
    "lalbagh botanical garden": (12.9507,77.5848),
    "cubbon park": (12.9763,77.5929),
    "bangalore palace": (12.9988,77.5921),
    "tipu sultan's summer palace": (12.9592,77.5733),
    "vidhana soudha (exterior viewpoint)": (12.9797,77.5907),
    "bannerghatta biological park": (12.8000,77.5770),
    "iskcon bengaluru": (13.0108,77.5511),
    "freedom park": (12.9789,77.5796),
    "government aquarium": (12.9744,77.5920),
    "nandi hills": (13.3702,77.6835),
    "mysuru palace": (12.3052,76.6552),
    "sri chamarajendra zoological gardens (mysuru zoo)": (12.3022,76.6644),
    "chamundi hills": (12.2724,76.6700),
    "jaganmohan palace art gallery": (12.3051,76.6529),
    "devaraja market": (12.3110,76.6544),
    "brindavan gardens": (12.4210,76.5720),
    "sand sculpture museum": (12.2947,76.6730),
    "st. philomena's church": (12.3250,76.6590),
    "karanji lake": (12.2958,76.6738),
    "mysuru rail museum": (12.3178,76.6575),
}

st.set_page_config(page_title="AI Travel Planner", page_icon="🧭", layout="wide")

@st.cache_data
def load_data():
    d = pd.read_csv(DATASET_FILE)
    required = {"destination","attraction_name","category","keywords","tags",
                "estimated_entry_fee_inr_per_person","estimated_visit_hours"}
    if required - set(d.columns):
        raise ValueError("CSV is missing columns: " + ", ".join(sorted(required-set(d.columns))))
    d["estimated_entry_fee_inr_per_person"] = pd.to_numeric(d["estimated_entry_fee_inr_per_person"], errors="coerce")
    d["estimated_visit_hours"] = pd.to_numeric(d["estimated_visit_hours"], errors="coerce")
    return d.dropna(subset=["destination","attraction_name","estimated_entry_fee_inr_per_person","estimated_visit_hours"])

def distance_km(a, b):
    pa, pb = COORDS.get(str(a).strip().casefold()), COORDS.get(str(b).strip().casefold())
    if not pa or not pb:
        return None
    lat1, lon1 = map(math.radians, pa)
    lat2, lon2 = map(math.radians, pb)
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    straight = 6371 * 2 * math.asin(math.sqrt(h))
    return straight * ROAD_FACTOR

def nearest_order(items):
    remaining = items[:]
    if len(remaining) < 2:
        return remaining
    ordered = [remaining.pop(0)]
    while remaining:
        prev = ordered[-1]["name"]
        idx = min(range(len(remaining)), key=lambda i: distance_km(prev, remaining[i]["name"]) if distance_km(prev, remaining[i]["name"]) is not None else 999999)
        ordered.append(remaining.pop(idx))
    return ordered

st.title("🧭 AI-Powered Personalized Travel Planner")
st.caption("Ollama + Llama 3.2 · dataset-grounded recommendations · route estimates · budget breakdown · evaluation")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load {DATASET_FILE}: {e}")
    st.stop()

if "eval_log" not in st.session_state:
    st.session_state.eval_log = []

@st.cache_data(show_spinner=False)
def get_cached_itinerary_response(prompt_text, model_name):
    """Cache the LLM response so saving an evaluation does not call Ollama again."""
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        format="json",
    )
    return response["message"]["content"]

with st.sidebar:
    st.header("Plan your trip")
    dests = sorted(df["destination"].astype(str).unique())
    destination = st.selectbox("Destination", dests)
    interests_text = st.text_input("Interests", "nature, history")
    request = st.text_area("Optional natural-language request", value=st.session_state.get("assistant_itinerary_request", ""), placeholder="Example: Plan a relaxing family trip with nature and photography.")
    days = st.number_input("Trip duration (days)", min_value=1, max_value=14, value=2)
    st.subheader("Budget estimates (per person)")
    entry_budget = st.number_input("Entry-fee budget (₹)", min_value=0, max_value=1000000, value=500, step=100)
    transport_day = st.number_input("Transport per day (₹)", min_value=0, max_value=100000, value=500, step=100)
    food_day = st.number_input("Food per day (₹)", min_value=0, max_value=100000, value=500, step=100)
    hotel_night = st.number_input("Accommodation per night (₹)", min_value=0, max_value=1000000, value=1500, step=250)
    generate = st.button("Generate my itinerary", type="primary", use_container_width=True)

st.info("Costs are estimates. Route distance/time uses approximate coordinates and a simple speed assumption—not live traffic or exact road directions.")
city = df[df.destination.astype(str).str.strip().str.casefold() == destination.casefold()].copy()
st.subheader(f"Explore {destination}")
st.write(f"**{len(city)} attractions** in the current dataset.")
with st.expander("Preview attractions in dataset"):
    st.dataframe(city[["attraction_name","category","estimated_entry_fee_inr_per_person","estimated_visit_hours"]], use_container_width=True, hide_index=True)

if generate:
    st.session_state["keep_itinerary_visible"] = True

if generate or st.session_state.get("keep_itinerary_visible", False):
    interests = [x.strip().lower() for x in interests_text.split(",") if x.strip()]
    user_request = request.strip()
    model_preferences = ", ".join(interests) if interests else "general sightseeing"
    if user_request:
        model_preferences += ". User request: " + user_request

    candidates = city[(city.estimated_entry_fee_inr_per_person >= 0) &
                      (city.estimated_visit_hours > 0) &
                      (city.estimated_entry_fee_inr_per_person <= float(entry_budget))].copy()
    if candidates.empty:
        st.warning("No attractions individually fit this entry-fee budget. Increase it or select another destination.")
        st.stop()

    catalog_cols = ["attraction_name","category","keywords","tags","estimated_entry_fee_inr_per_person","estimated_visit_hours"]
    catalog = candidates[catalog_cols].to_dict(orient="records")
    preferred_names = [
        name for name in st.session_state.get("assistant_itinerary_names", [])
        if name.casefold() in {str(x).strip().casefold() for x in candidates["attraction_name"]}
    ]
    preferred_instruction = (
        f"Prioritize these validated assistant recommendations if relevant and affordable: {json.dumps(preferred_names)}."
        if preferred_names else "Choose attractions from the catalog based on the user's interests."
    )
    prompt = f"""Create a {int(days)}-day itinerary for {destination}.
Preferences: {model_preferences}
Total entry-fee budget per person: INR {float(entry_budget):.0f}.
{preferred_instruction}
Use ONLY attractions in this catalog: {json.dumps(catalog, ensure_ascii=False)}
Return JSON only: {{"itinerary":[{{"day":1,"attractions":[{{"attraction_name":"exact catalog name","reason":"why it matches"}}]}}]}}
Use exact names only; no invented places/prices/durations; no duplicates; day numbers 1 to {int(days)}.
Aim to schedule at least one suitable attraction on every day when enough affordable attractions exist.
Prefer 1 to 3 attractions per day, but do not exceed the daily visit-time limit.
Total entry fees across the whole itinerary must fit the budget."""
    with st.spinner("Generating itinerary with Llama 3.2..."):
        try:
            response_text = get_cached_itinerary_response(prompt, MODEL)
            model_result = json.loads(response_text)
        except Exception as e:
            st.error("Ollama could not generate the itinerary. Ensure Ollama is running and llama3.2 is installed.")
            st.code(str(e))
            st.stop()

    by_name = {str(r.attraction_name).strip().casefold(): r for _,r in candidates.iterrows()}
    planned = {day: [] for day in range(1,int(days)+1)}
    notes = []
    used = set()
    planned_entry_cost = 0.0
    raw_days = model_result.get("itinerary", [])
    if not isinstance(raw_days,list):
        raw_days = []
        notes.append("Model returned an invalid itinerary structure.")

    for day_item in raw_days:
        if not isinstance(day_item,dict):
            notes.append("Skipped malformed day.")
            continue
        try:
            day = int(day_item.get("day"))
        except (ValueError,TypeError):
            notes.append("Skipped day with invalid number.")
            continue
        if day not in planned:
            notes.append(f"Skipped invalid day number {day}.")
            continue
        attractions = day_item.get("attractions",[])
        if not isinstance(attractions,list):
            continue
        for item in attractions:
            if not isinstance(item,dict):
                continue
            name = str(item.get("attraction_name","")).strip()
            key = name.casefold()
            if key not in by_name:
                notes.append(f"Rejected unknown attraction: {name or '(blank)'}.")
                continue
            if key in used:
                notes.append(f"Rejected duplicate attraction: {name}.")
                continue
            row = by_name[key]
            fee = float(row.estimated_entry_fee_inr_per_person)
            if planned_entry_cost + fee > float(entry_budget):
                notes.append(f"Rejected {name}: adding its entry fee would exceed the total entry-fee budget.")
                continue
            planned[day].append({"name":str(row.attraction_name),"category":str(row.category),
                                 "fee":fee,
                                 "hours":float(row.estimated_visit_hours),
                                 "reason":str(item.get("reason","")).strip() or "Selected from the dataset."})
            planned_entry_cost += fee
            used.add(key)

    # Python fallback: if the LLM leaves a day empty, fill it with the best
    # remaining CSV attraction matching the user's interests, subject to budget.
    # This makes day allocation less dependent on the LLM returning perfect JSON.
    interest_terms = [term.strip().casefold() for term in interests if term.strip()]
    for empty_day in planned:
        if planned[empty_day]:
            continue

        remaining = candidates[
            ~candidates["attraction_name"].astype(str).str.strip().str.casefold().isin(used)
        ].copy()
        if remaining.empty:
            notes.append(f"Day {empty_day} remains empty: no unused attractions are available in the dataset.")
            continue

        def fallback_score(row):
            searchable = " ".join([
                str(row.get("category", "")),
                str(row.get("keywords", "")),
                str(row.get("tags", "")),
                str(row.get("attraction_name", "")),
            ]).casefold()
            return sum(1 for term in interest_terms if term in searchable)

        remaining["_match_score"] = remaining.apply(fallback_score, axis=1)
        remaining = remaining.sort_values(
            by=["_match_score", "estimated_visit_hours"],
            ascending=[False, True]
        )

        chosen = None
        for _, candidate_row in remaining.iterrows():
            candidate_name = str(candidate_row["attraction_name"]).strip()
            candidate_key = candidate_name.casefold()
            candidate_fee = float(candidate_row["estimated_entry_fee_inr_per_person"])
            if planned_entry_cost + candidate_fee <= float(entry_budget):
                chosen = candidate_row
                break

        if chosen is None:
            notes.append(f"Day {empty_day} remains empty: no unused attraction fits the remaining entry-fee budget.")
            continue

        chosen_name = str(chosen["attraction_name"]).strip()
        planned[empty_day].append({
            "name": chosen_name,
            "category": str(chosen["category"]),
            "fee": float(chosen["estimated_entry_fee_inr_per_person"]),
            "hours": float(chosen["estimated_visit_hours"]),
            "reason": "Added by Python fallback to fill an empty day, using dataset details and your interests."
        })
        planned_entry_cost += float(chosen["estimated_entry_fee_inr_per_person"])
        used.add(chosen_name.casefold())
        notes.append(f"Day {empty_day} was filled by Python fallback using a remaining dataset attraction.")

    # Reorder each day's stops to reduce estimated between-stop distance.
    # Then enforce daily visiting + travel time; skipped stops are disclosed.
    scheduled = {day: [] for day in planned}
    for day in planned:
        ordered = nearest_order(planned[day])
        elapsed = 0.0
        previous = None
        for item in ordered:
            dist = distance_km(previous["name"], item["name"]) if previous else 0.0
            travel = (dist / AVG_SPEED_KMH) if dist is not None else 0.0
            if elapsed + item["hours"] + travel > MAX_DAY_HOURS:
                notes.append(f"Skipped {item['name']}: estimated visits plus travel would exceed {MAX_DAY_HOURS} hours on Day {day}.")
                continue
            scheduled[day].append({**item,"distance_km":dist,"travel_hours":travel})
            elapsed += item["hours"] + travel
            previous = item

    accepted = [x for day in scheduled.values() for x in day]
    total_entry = sum(x["fee"] for x in accepted)
    total_transport = float(transport_day) * int(days)
    total_food = float(food_day) * int(days)
    nights = max(int(days)-1,0)
    total_hotel = float(hotel_night) * nights
    total_trip = total_entry + total_transport + total_food + total_hotel
    total_distance = sum(x["distance_km"] or 0 for x in accepted)
    total_travel = sum(x["travel_hours"] for x in accepted)
    entry_pass = total_entry <= float(entry_budget)
    names_pass = all(x["name"].casefold() in by_name for x in accepted)
    duplicate_pass = len({x["name"].casefold() for x in accepted}) == len(accepted)
    schedule_pass = all(sum(x["hours"]+x["travel_hours"] for x in scheduled[d]) <= MAX_DAY_HOURS+1e-9 for d in scheduled)

    st.subheader("Your personalized itinerary")
    a,b,c = st.columns(3)
    a.metric("Validated attractions",len(accepted))
    b.metric("Estimated total trip cost",f"₹{total_trip:,.0f}")
    c.metric("Entry budget remaining",f"₹{float(entry_budget)-total_entry:,.0f}")

    st.markdown("### Budget breakdown")
    costs = pd.DataFrame([
        {"Category":"Attraction entry fees","Estimated cost (₹)":total_entry},
        {"Category":"Transport (daily allowance × days)","Estimated cost (₹)":total_transport},
        {"Category":"Food (daily allowance × days)","Estimated cost (₹)":total_food},
        {"Category":f"Accommodation ({nights} night(s))","Estimated cost (₹)":total_hotel},
        {"Category":"TOTAL ESTIMATED TRIP COST","Estimated cost (₹)":total_trip},
    ])
    st.dataframe(costs,use_container_width=True,hide_index=True)

    for day, entries in scheduled.items():
        with st.expander(f"Day {day} — {len(entries)} attraction(s)",expanded=True):
            if not entries:
                st.write("No attractions assigned.")
                continue
            for i,x in enumerate(entries):
                st.markdown(f"**{i+1}. {x['name']}** · {x['category']}")
                st.write(x["reason"])
                st.caption(f"Entry estimate ₹{x['fee']:.0f} · Visit {x['hours']:.1f} h")
                if i > 0 and x["distance_km"] is not None:
                    st.caption(f"Approx. from previous stop: {x['distance_km']:.1f} km · travel {x['travel_hours']:.1f} h")
            st.write(f"Day estimate: ₹{sum(x['fee'] for x in entries):.0f} entry fees · "
                     f"{sum(x['hours'] for x in entries):.1f} h visits · "
                     f"{sum(x['travel_hours'] for x in entries):.1f} h estimated travel")

    st.caption(f"Approx. between-attraction distance: {total_distance:.1f} km; travel time: {total_travel:.1f} h. This is not live navigation.")
    st.warning("The entry-fee budget is checked separately. The full trip total adds your transport, food and accommodation allowances. Accommodation assumes days − 1 nights.")
    if notes:
        with st.expander("Validation and scheduling notes"):
            for note in notes:
                st.write("• "+note)

    def evaluation_panel():
        st.markdown("### Evaluate this test run")
        with st.form("evaluation_form"):
            relevance = st.slider(
                "Human relevance rating (1 = poor, 5 = excellent)",
                1, 5, 3
            )
            test_notes = st.text_input(
                "Test notes (optional)",
                placeholder="e.g. relevant places, unrealistic travel estimate"
            )
            save_evaluation = st.form_submit_button("Save evaluation result")

        if save_evaluation:
            st.session_state.eval_log.append({
                "timestamp":datetime.now().isoformat(timespec="seconds"),
                "destination":destination,"interests":interests_text,"natural_request":request,
                "days":int(days),"entry_budget_inr":float(entry_budget),
                "transport_per_day_inr":float(transport_day),"food_per_day_inr":float(food_day),
                "accommodation_per_night_inr":float(hotel_night),
                "accepted_attractions":len(accepted),"estimated_entry_fees_inr":total_entry,
                "estimated_total_trip_cost_inr":total_trip,"valid_names_pass":names_pass,
                "no_duplicates_pass":duplicate_pass,"entry_budget_pass":entry_pass,
                "daily_schedule_pass":schedule_pass,"human_relevance_rating":relevance,"notes":test_notes
            })
            st.success("Evaluation saved. Download the CSV below to keep a permanent copy.")

        if st.session_state.eval_log:
            log_df = pd.DataFrame(st.session_state.eval_log)
            st.markdown("### Evaluation log")
            st.dataframe(log_df,use_container_width=True,hide_index=True)
            st.download_button("Download evaluation results (CSV)",log_df.to_csv(index=False).encode("utf-8"),
                               file_name="travel_planner_evaluation_results.csv",mime="text/csv")

    evaluation_panel()

    export = {
        "destination":destination,"interests":interests_text,"natural_language_request":request,
        "days":int(days),"budget_breakdown_inr":{"entry_fees":total_entry,"transport":total_transport,
        "food":total_food,"accommodation":total_hotel,"total_trip":total_trip},
        "itinerary":[{"day":day,"attractions":[{"name":x["name"],"category":x["category"],
        "reason":x["reason"],"entry_fee_inr":x["fee"],"visit_hours":x["hours"],
        "approx_distance_from_previous_km":x["distance_km"],"approx_travel_hours":x["travel_hours"]}
        for x in scheduled[day]]} for day in scheduled]
    }
    st.download_button("Download itinerary (JSON)",json.dumps(export,indent=2,ensure_ascii=False),
                       file_name="travel_itinerary.json",mime="application/json")


# AI travel assistant: text chat, optional speech input, and browser voice output.
st.divider()
st.header("💬 AI Travel Assistant")
st.write(
    "Ask questions about attractions in this dataset, your itinerary, estimated costs, "
    "what to pack, or how to plan your days. You can type or use your microphone."
)

if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
if "assistant_last_answer" not in st.session_state:
    st.session_state.assistant_last_answer = ""

# Optional browser speech recognition. Install streamlit-mic-recorder to enable it.
voice_text = ""
try:
    from streamlit_mic_recorder import speech_to_text
    voice_text = speech_to_text(
        language="en",
        start_prompt="🎙️ Speak your question",
        stop_prompt="⏹️ Stop recording",
        just_once=True,
        key="travel_assistant_speech_input",
    ) or ""
except ImportError:
    st.info("To enable microphone input, run: python -m pip install streamlit-mic-recorder")
except Exception as speech_error:
    st.caption(f"Microphone input is unavailable in this browser/session: {speech_error}")

if voice_text:
    st.session_state.assistant_voice_question = voice_text

if st.session_state.get("assistant_voice_question"):
    st.caption("Recognized speech:")
    st.write(st.session_state.assistant_voice_question)

typed_question = st.text_input(
    "Your question",
    placeholder="e.g. Which Bengaluru attractions are good for nature lovers?",
    key="assistant_typed_question",
)
col_send, col_voice, col_clear = st.columns([1, 1, 1])
with col_send:
    send_typed = st.button("Ask assistant", type="primary", use_container_width=True)
with col_voice:
    send_voice = st.button("Ask using recognized speech", use_container_width=True)
with col_clear:
    clear_chat = st.button("Clear chat", use_container_width=True)

if clear_chat:
    st.session_state.assistant_messages = []
    st.session_state.assistant_last_answer = ""
    st.session_state.assistant_voice_question = ""
    st.rerun()

question = ""
if send_typed:
    question = typed_question.strip()
elif send_voice:
    question = str(st.session_state.get("assistant_voice_question", "")).strip()

# Render saved conversation before processing a new question.
# A newly streamed answer is displayed separately, so it appears only once.
for message in st.session_state.assistant_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:
    # Give the assistant a compact, grounded reference from the current CSV.
    # Keep the prompt small for faster local inference: use the selected destination only.
    context_df = df[
        df["destination"].astype(str).str.strip().str.casefold()
        == destination.casefold()
    ][[
        "destination", "attraction_name", "category", "keywords", "tags",
        "estimated_entry_fee_inr_per_person", "estimated_visit_hours"
    ]].copy()
    is_indoor_query = any(
        term in question.casefold()
        for term in ["indoor", "inside", "rainy", "raining", "rain", "bad weather"]
    )
    if is_indoor_query:
        indoor_mask = context_df["tags"].fillna("").astype(str).apply(
            lambda value: "indoor" in [tag.strip().casefold() for tag in value.split(";")]
        )
        context_df = context_df[indoor_mask].drop_duplicates(
            subset=["attraction_name"], keep="first"
        )
        if context_df.empty:
            answer = (
                f"The {destination} dataset has no attractions explicitly tagged 'indoor'. "
                "I don't want to guess or label outdoor places as indoor. Add verified indoor attractions to the CSV."
            )
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.assistant_messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            st.session_state.assistant_last_answer = answer
            st.session_state.assistant_voice_question = ""
            st.stop()

    context = context_df.to_dict(orient="records")
    recent_history = st.session_state.assistant_messages[-4:]
    system_prompt = f"""
You are the AI assistant inside an academic travel-planner prototype.
Answer the user's travel-planning questions clearly and simply.
Use this dataset as the source of truth for attraction names, categories, entry-fee estimates,
and visit durations:
{json.dumps(context, ensure_ascii=False)}

Rules:
- Do not invent dataset entries, prices, opening hours, travel times, weather, or availability.
- Prices and durations are estimates, not live-verified facts.
- If the dataset does not contain the answer, say so clearly and label any general knowledge separately.
- For indoor/rain questions, use ONLY the supplied records explicitly tagged indoor. If there is only one, recommend only one; never pad the list with duplicates or outdoor places.
- Mention each attraction at most once. Do not invent extra places to make a longer list.
- Do not claim a place is open today or that it is currently raining unless a live data source has actually been connected.
- For cost questions, distinguish entry fees from the full trip estimate.
- Keep answers helpful and concise.
- If the user asks for an itinerary, suggest only attractions in the supplied dataset and never invent prices.
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": question})

    try:
        # Stream tokens as they are generated, so the user sees progress immediately.
        with st.chat_message("assistant"):
            streamed_answer = st.write_stream(
                chunk["message"]["content"]
                for chunk in ollama.chat(
                    model=MODEL,
                    messages=messages,
                    stream=True,
                    options={"num_ctx": 2048, "num_predict": 220},
                )
                if chunk.get("message", {}).get("content")
            )
        answer = str(streamed_answer).strip()
        # Extra safeguard: remove repeated lines that mention an attraction already listed.
        known_names = [str(n) for n in context_df["attraction_name"].dropna().unique()]
        seen_names = set()
        cleaned_lines = []
        for line in answer.splitlines():
            matches = [n for n in known_names if n.casefold() in line.casefold()]
            if matches and any(n.casefold() in seen_names for n in matches):
                continue
            cleaned_lines.append(line)
            seen_names.update(n.casefold() for n in matches)
        answer = "\n".join(cleaned_lines).strip()
        st.session_state.assistant_messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        st.session_state.assistant_last_answer = answer
        st.session_state.assistant_voice_question = ""
    except Exception as assistant_error:
        st.error("Could not contact Ollama. Make sure Ollama is running and the llama3.2 model is installed.")
        st.code(str(assistant_error))

if st.session_state.assistant_last_answer:
    try:
        import streamlit.components.v1 as components
        speak_text = json.dumps(st.session_state.assistant_last_answer)
        components.html(
            f"""
            <div style="font-family: sans-serif; padding: 4px 0;">
              <button onclick="speakAnswer()" style="padding:8px 12px; cursor:pointer;">
                🔊 Read the latest answer aloud
              </button>
              <button onclick="window.speechSynthesis.cancel()" style="padding:8px 12px; cursor:pointer;">
                Stop speech
              </button>
            </div>
            <script>
              const answerText = {speak_text};
              function speakAnswer() {{
                if (!('speechSynthesis' in window)) {{
                  alert('Speech output is not supported in this browser.');
                  return;
                }}
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(answerText);
                utterance.lang = 'en-IN';
                utterance.rate = 1;
                window.speechSynthesis.speak(utterance);
              }}
            </script>
            """,
            height=55,
        )
    except Exception:
        st.caption("Voice playback may not be supported in this browser.")

    if st.button("🗺️ Use this answer to plan my itinerary"):
        latest_answer = st.session_state.assistant_last_answer
        selected_city = df[
            df["destination"].astype(str).str.strip().str.casefold() == destination.casefold()
        ]
        # Only pass exact attraction names that exist in the selected destination's CSV.
        validated_names = [
            name for name in selected_city["attraction_name"].astype(str).drop_duplicates()
            if name.casefold() in latest_answer.casefold()
        ]
        st.session_state["assistant_itinerary_names"] = validated_names
        st.session_state["assistant_itinerary_request"] = (
            "Use these dataset-validated assistant recommendations as preferences: "
            + (", ".join(validated_names) if validated_names else "No exact attraction names were validated.")
            + ". Use only attractions in the selected destination dataset, check the total entry-fee budget, "
              "and do not repeat attractions."
        )
        st.success(
            f"Validated {len(validated_names)} attraction name(s). "
            "Click 'Generate my itinerary' in the sidebar; the names and budget will be checked again."
        )
        st.rerun()


st.divider()
st.caption("Academic prototype. Prices/durations are dataset estimates; route calculations use approximate coordinates and assumptions. Verify real prices, opening hours and travel conditions.")
