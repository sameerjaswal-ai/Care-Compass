import re

import pandas as pd
import streamlit as st

from nlp import analyze_query
from recommendation import compare_hospitals, recommend_hospitals


HEALTHCARE_TERMS = [
    "hospital",
    "hospitals",
    "clinic",
    "doctor",
    "doctors",
    "treatment",
    "treatments",
    "patient",
    "care",
    "medical",
    "health",
    "healthcare",
    "surgery",
    "emergency",
    "recommend",
    "recommendation",
    "compare",
    "versus",
    "rating",
    "success",
    "beds",
    "specialist",
    "specialty",
    "speciality",
    "specialties",
    "nabh",
]

OFF_TOPIC_TERMS = {
    "car",
    "cars",
    "vehicle",
    "vehicles",
    "bike",
    "bikes",
    "motorcycle",
    "bus",
    "train",
    "flight",
    "flights",
    "hotel",
    "hotels",
    "movie",
    "movies",
    "laptop",
    "phone",
    "mobile",
    "restaurant",
    "shopping",
    "clothes",
    "school",
    "college",
    "job",
    "jobs",
    "cricket",
    "football",
    "stock",
    "stocks",
}


def initialize_chatbot():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_chat_history():
    initialize_chatbot()

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi, I'm the **CareCompass Hospital Assistant**. "
                "Ask me to find or compare hospitals by disease, city, "
                "or specialty. I always rank results by success rate."
            )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _cell(row, *column_names):
    for column in column_names:
        if column in row.index:
            value = row[column]
            if pd.notna(value):
                return value
    return None


def is_relevant_query(query, query_info):
    lowered = str(query).lower()
    tokens = set(re.findall(r"[a-z0-9]+", lowered))

    if tokens & OFF_TOPIC_TERMS:
        return False

    has_medical_term = any(
        re.search(r"\b" + re.escape(term) + r"\b", lowered)
        for term in HEALTHCARE_TERMS
    )
    has_disease = bool(query_info.get("disease"))
    has_specialty = bool(query_info.get("specialty"))
    has_hospital = bool(query_info.get("hospital") or query_info.get("hospitals"))

    if has_medical_term or has_disease or has_specialty or has_hospital:
        return True

    return False


def off_topic_response():
    return (
        "I can only help with **hospital recommendations and comparisons**. "
        "Queries like cars, shopping, or travel are outside what I can do.\n\n"
        "Try something like:\n"
        "- `dengue hospitals in Punjab`\n"
        "- `cardiology hospitals in Delhi under 50000`\n"
        "- `compare AIIMS Delhi and Sir Ganga Ram Hospital`"
    )


def format_hospital_card(row, index):
    name = _cell(row, "hospital_name", "name") or "Unknown hospital"
    response = f"**{index}. {name}**\n\n"

    city = _cell(row, "city")
    state = _cell(row, "state")
    if city or state:
        location = ", ".join(part for part in [city, state] if part)
        response += f"📍 **Location:** {location}\n\n"

    hospital_type = _cell(row, "hospital_type")
    if hospital_type:
        response += f"🏥 **Type:** {hospital_type}\n\n"

    rating = pd.to_numeric(_cell(row, "rating"), errors="coerce")
    if pd.notna(rating):
        response += f"⭐ **Rating:** {rating:.1f}\n\n"

    cost = pd.to_numeric(_cell(row, "treatment_cost_inr"), errors="coerce")
    if pd.notna(cost):
        response += f"💰 **Treatment Cost:** ₹{cost:,.0f}\n\n"

    success_rate = pd.to_numeric(_cell(row, "success_rate"), errors="coerce")
    if pd.notna(success_rate):
        response += f"📊 **Success Rate:** {success_rate:.2f}%\n\n"

    disease = _cell(row, "disease")
    if disease:
        response += f"🦠 **Disease:** {disease}\n\n"

    specialties = _cell(row, "specialties")
    if specialties:
        response += f"🩺 **Specialties:** {specialties}\n\n"

    response += "---\n\n"
    return response


def format_hospital_results(results, query_info):
    if results is None or results.empty:
        return (
            "I couldn't find any hospitals matching your requirements. "
            "Try another city, disease, specialty, or budget."
        )

    hints = []
    if query_info.get("disease"):
        hints.append(f"disease **{query_info['disease']}**")
    if query_info.get("specialty"):
        hints.append(f"specialty **{query_info['specialty']}**")
    if query_info.get("city"):
        hints.append(f"city **{query_info['city']}**")
    if query_info.get("state"):
        hints.append(f"state **{query_info['state']}**")
    if query_info.get("budget") is not None:
        hints.append(f"budget up to ₹{query_info['budget']:,.0f}")

    response = "### Recommended Hospitals\n\n"
    response += "Sorted by **success rate** (highest first).\n\n"
    if hints:
        response += "Matching " + ", ".join(hints) + ":\n\n"

    for index, row in results.iterrows():
        response += format_hospital_card(row, index + 1)

    return response


def format_comparison(results, hospital_names):
    if results is None or results.empty:
        names = " and ".join(f"**{name}**" for name in hospital_names)
        return (
            f"I couldn't compare {names}. "
            "Please use hospital names from the dataset."
        )

    response = "### Hospital Comparison\n\n"
    response += "Sorted by **success rate** (highest first).\n\n"

    if len(results) == 1:
        found = _cell(results.iloc[0], "hospital_name") or "one hospital"
        response += (
            f"I only found **{found}**. "
            "Name two hospitals to compare.\n\n"
        )

    for index, row in results.iterrows():
        response += format_hospital_card(row, index + 1)

    if len(results) >= 2:
        top = _cell(results.iloc[0], "hospital_name") or "the first hospital"
        top_rate = pd.to_numeric(
            _cell(results.iloc[0], "success_rate"),
            errors="coerce",
        )
        if pd.notna(top_rate):
            response += (
                f"**{top}** has the higher success rate "
                f"({top_rate:.2f}%)."
            )
        else:
            response += f"**{top}** ranks first in this comparison."

    return response


def process_query(query, hospital_data):
    query_info = analyze_query(query, hospital_data)
    query_info["sort_by"] = "success_rate"

    if not is_relevant_query(query, query_info):
        return off_topic_response()

    if query_info.get("intent") == "compare":
        hospital_names = query_info.get("hospitals") or []
        if len(hospital_names) < 2:
            return (
                "To compare hospitals, name **two hospitals**, for example: "
                "`compare AIIMS Delhi and Sir Ganga Ram Hospital`."
            )

        results = compare_hospitals(hospital_data, hospital_names[:2])
        return format_comparison(results, hospital_names[:2])

    results = recommend_hospitals(hospital_data, query_info)
    return format_hospital_results(results, query_info)


def run_chatbot(hospital_data):
    initialize_chatbot()

    st.markdown("---")
    with st.expander(
        "Ask CareCompass about hospitals",
        expanded=bool(st.session_state.messages),
    ):
        st.caption(
            "Ask about hospitals by disease, location, or specialty, "
            "or compare two hospitals. Results are ranked by success rate."
        )
        display_chat_history()

    user_query = st.chat_input("Ask me about hospitals...")

    if not user_query:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
    })

    try:
        response = process_query(user_query, hospital_data)
    except Exception:
        response = (
            "Something went wrong while searching hospitals. "
            "Please try a hospital-related query, for example: "
            "`dengue hospitals in Delhi` or "
            "`compare AIIMS Delhi and Sir Ganga Ram Hospital`."
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
    })
    st.rerun()
