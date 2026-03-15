#  Must be first!
import streamlit as st
st.set_page_config(page_title="Smart Travel Advisor", layout="wide")

import pandas as pd
import os
from typing import Dict, Any


# ------------------------- API Configuration -------------------------
os.environ["RECIPE_GROQ_API_KEY"] = "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"
os.environ["HF_EMBEDDINGS_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
#  Updated Groq model (the old one was decommissioned)
os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
os.environ["OPENAI_API_KEY"] = "sk-yourapikey"
os.environ["DATABASE_URL"] = "sqlite:///./resumes.db"

# ------------------------- LangChain / Groq Imports -------------------------
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS, Chroma
    from langchain_core.documents import Document
    from langchain_groq import ChatGroq
    LANGCHAIN_AVAILABLE = True

    #  Test Groq connectivity
    try:
        llm_test = ChatGroq(
            groq_api_key=os.environ["RECIPE_GROQ_API_KEY"],
            model=os.environ["GROQ_MODEL"]
        )
        _ = llm_test.invoke("ping")
        #st.success(f" Groq connected successfully using model: {os.environ['GROQ_MODEL']}")
    except Exception as e:
        LANGCHAIN_AVAILABLE = False
        st.warning(f" Groq connection failed: {e}")

except Exception as e:
    LANGCHAIN_AVAILABLE = False
    st.warning(f" LangChain or Groq not fully available: {e}")

# ------------------------- Helper utilities -------------------------
@st.cache_data
def load_destinations(csv_path: str = "smart_travel_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        alt = "master_travel_dataset_20000.csv"
        if os.path.exists(alt):
            csv_path = alt
            st.info(f"Loaded large dataset: {alt}")
        else:
            st.warning("No dataset found — place smart_travel_dataset.csv or master_travel_dataset_20000.csv in the app folder.")
            return pd.DataFrame()

    df = pd.read_csv(csv_path, encoding='utf-8')
    df = df.rename(columns=lambda c: c.strip())

    def get_col(*names, default="Not specified"):
        for n in names:
            if n in df.columns:
                return df[n].fillna(default)
        return pd.Series([default] * len(df))

    mapped = pd.DataFrame()
    mapped['name'] = get_col('City', 'city', 'Name', 'city_name')
    mapped['country'] = get_col('Country', 'country')
    mapped['region'] = get_col('Region', 'region')
    mapped['category'] = get_col('Category', 'category')
    mapped['ideal_climate'] = get_col('Climate', 'climate')
    mapped['best_months'] = get_col('Best_Month', 'best_months')
    mapped['attractions'] = get_col('Attractions', 'attractions', 'Attraction')
    mapped['cuisine'] = get_col('Cuisine_Name', 'Cuisine', 'cuisine_name')
    mapped['flavor_profile'] = get_col('Flavor_Profile', 'flavor_profile')
    mapped['hotel_name'] = get_col('Hotel_Name', 'Hotel')
    mapped['activities'] = get_col('Activities', 'activities')
    mapped['notes'] = get_col('Notes', 'notes')
    mapped['amenities'] = get_col('Amenities', 'amenities')
    mapped['average_temp_c'] = get_col('Average_Temperature_C', 'avg_temp_c')

    if 'Price_Per_Night_USD' in df.columns:
        mapped['estimated_cost_per_person_per_day'] = df['Price_Per_Night_USD'].fillna(0)
    elif 'Estimated_Budget_5days_USD' in df.columns:
        mapped['estimated_cost_per_person_per_day'] = (df['Estimated_Budget_5days_USD'] / 5.0).fillna(0)
    elif 'Price_Per_Night' in df.columns:
        mapped['estimated_cost_per_person_per_day'] = df['Price_Per_Night'].fillna(0)
    else:
        mapped['estimated_cost_per_person_per_day'] = pd.Series([0] * len(df))

    mapped.index = df.index
    return mapped


def score_destination(row: pd.Series, prefs: Dict[str, Any]) -> float:
    score = 0.0
    if prefs.get('preferred_climate') and prefs['preferred_climate'].lower() in str(row.get('ideal_climate', '')).lower():
        score += 2.0
    if prefs.get('destination_type') and prefs['destination_type'].lower() in str(row.get('category', '')).lower():
        score += 2.5

    est = float(row.get('estimated_cost_per_person_per_day') or 0)
    user_daily_budget = ((prefs.get('budget_min', 0) + prefs.get('budget_max', 0)) / 2.0) / max(1, int(prefs.get('duration_days', 1)))
    if user_daily_budget > 0:
        budget_diff = abs(est - user_daily_budget)
        score += max(0, 2.0 - (budget_diff / max(100.0, user_daily_budget)))

    if prefs.get('travel_month') and pd.notna(row.get('best_months')):
        bm = str(row.get('best_months')).lower()
        if prefs['travel_month'].lower() in bm:
            score += 1.5

    interests = prefs.get('interests') or []
    combined_text = " ".join([str(row.get('attractions', '')), str(row.get('activities', '')), str(row.get('notes', ''))]).lower()
    for it in interests:
        if it.lower() in combined_text:
            score += 1.0
    return score


def heuristic_recommend(df: pd.DataFrame, prefs: Dict[str, Any], top_k: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d['score'] = d.apply(lambda r: score_destination(r, prefs), axis=1)
    d = d.sort_values('score', ascending=False)
    return d.head(top_k)

# ------------------------- LLM itinerary generation -------------------------
def llm_generate_itinerary(destination: Dict[str, Any], prefs: Dict[str, Any]) -> str:
    interests_text = ', '.join(prefs.get('interests', [])) or 'general activities'
    days_count = int(prefs.get('duration_days', 1))
    system_prompt = f"You are an expert travel planner. Create a {days_count}-day itinerary for a traveler who wants {interests_text} in {destination.get('name', 'Unknown')}, {destination.get('country', '')}."

    details = [
        f"Destination: {destination.get('name', '')}",
        f"Category: {destination.get('category', '')}",
        f"Attractions: {destination.get('attractions', '')}",
        f"Cuisine: {destination.get('cuisine', '')}",
        f"Estimated cost/day (USD): {destination.get('estimated_cost_per_person_per_day', '')}",
        f"User preferences: climate={prefs.get('preferred_climate', '')}, month={prefs.get('travel_month', '')}, budget_range=${prefs.get('budget_min', '')}-${prefs.get('budget_max', '')}, interests={prefs.get('interests', [])}"
    ]
    prompt = system_prompt + "\n\nDetails:\n" + "\n".join(details)

    if LANGCHAIN_AVAILABLE and os.getenv("RECIPE_GROQ_API_KEY"):
        try:
            llm = ChatGroq(
                groq_api_key=os.environ["RECIPE_GROQ_API_KEY"],
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.7
            )
            result = llm.invoke(prompt)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            st.error(f"Groq LLM call failed: {e}. Using offline fallback.")

    est = int(float(destination.get('estimated_cost_per_person_per_day') or 100))
    lines = [f"Summary: {destination.get('name', 'This destination')} offers {destination.get('category', 'diverse')} experiences and matches interests in {interests_text}."]
    for d in range(1, days_count + 1):
        lines.append(f"Day {d}: Morning - explore {destination.get('attractions', 'local attractions')}. Afternoon - try {destination.get('cuisine', 'local food')}. Evening - relax. Approx cost: ${est}")
    lines.append("Hotel options: Budget guesthouse, Mid-range hotel, Luxury resort.")
    return "\n".join(lines)

# ------------------------- Streamlit UI -------------------------
def main():
    st.title("Smart Travel Destination Advisor and Planner")

    if 'assistant_history' not in st.session_state:
        st.session_state['assistant_history'] = []
    if 'last_generated_plan' not in st.session_state:
        st.session_state['last_generated_plan'] = None

    st.sidebar.header("Your Preferences")
    preferred_climate = st.sidebar.selectbox("Preferred climate", ['', 'Hot', 'Cold', 'Moderate', 'Humid', 'Cool'])
    destination_type = st.sidebar.selectbox("Type of destination", ['', 'Beach', 'Mountain', 'City', 'Forest', 'Heritage', 'Island', 'Desert'])
    budget_range = st.sidebar.slider("Budget range (USD) total per person", 100, 10000, (1000, 5000), step=50)
    duration_days = st.sidebar.number_input("Duration (days)", min_value=1, max_value=30, value=5)
    travel_month = st.sidebar.text_input("Travel month or season (e.g., Jan, Dec, winter)")
    interests = st.sidebar.multiselect("Interests", ['Adventure', 'Relaxation', 'Food', 'Culture', 'Wildlife', 'Photography'])

    df = load_destinations()
    if df.empty:
        st.info("No dataset loaded. Put master_travel_dataset_20000.csv in the app folder and refresh.")
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        st.header("Find destinations")
        if st.button("Recommend"):
            prefs = {
                'preferred_climate': preferred_climate,
                'destination_type': destination_type,
                'budget_min': budget_range[0],
                'budget_max': budget_range[1],
                'duration_days': int(duration_days),
                'travel_month': travel_month,
                'interests': interests
            }
            candidates = heuristic_recommend(df, prefs, top_k=10)

            if candidates.empty:
                st.warning("No matching destinations found.")
            else:
                for idx, row in candidates.iterrows():
                    st.markdown(f"### {row.get('name', 'Unknown')} — {row.get('country', '')}")
                    st.write(f"Category: {row.get('category', '')} | Climate: {row.get('ideal_climate', '')}")
                    if st.button(f"Plan trip to {row.get('name', '')}", key=f"plan_{idx}"):

                        with st.spinner("Generating itinerary..."):
                            plan = llm_generate_itinerary(row.to_dict(), prefs)
                            st.session_state['last_generated_plan'] = {'destination': row.to_dict(), 'plan': plan}
                            st.subheader(f"Plan for {row.get('name', '')}")
                            st.write(plan)

    with col2:
        st.header("Interactive Travel Assistant")
        user_query = st.text_area("Ask your assistant")
        if st.button("Submit query") and user_query.strip():
            if st.session_state.get('last_generated_plan'):
                prompt = f"Modify the following travel plan:\n\n{st.session_state['last_generated_plan']['plan']}\n\nUser request: {user_query}"
            else:
                prompt = f"Create a travel plan considering: {user_query}"

            if LANGCHAIN_AVAILABLE and os.getenv("RECIPE_GROQ_API_KEY"):
                try:
                    llm = ChatGroq(
                        groq_api_key=os.environ["RECIPE_GROQ_API_KEY"],
                        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                        temperature=0.7
                    )
                    result = llm.invoke(prompt)
                    resp = result.content if hasattr(result, "content") else str(result)
                except Exception as e:
                    resp = f"(Groq API call failed: {e})"
            else:
                resp = "(Offline mode) Please enable Groq API key to use AI refinement."

            st.session_state['assistant_history'].append({'query': user_query, 'response': resp})

        for item in reversed(st.session_state.get('assistant_history', [])[-5:]):
            st.markdown(f"**Q:** {item['query']}\n\n**A:** {item['response']}")

    st.markdown("---")
    #st.caption(" Built with Streamlit + LangChain + Groq + Hugging Face embeddings.")

if __name__ == '__main__':
    main()
