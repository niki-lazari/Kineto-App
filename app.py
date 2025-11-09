import streamlit as st
from PIL import Image
import base64
import time
import threading
from AllModelsTrial import get_recommender
#from capstonemasterv3 import get_recommender


@st.cache_resource
def get_cached_recommender():
    """Loads and caches the recommender once."""
    return get_recommender()

def preload_recommender():
    """Triggers the cache load in background."""
    get_cached_recommender()

# Start background thread as soon as app starts
if 'recommender_preloading' not in st.session_state:
    st.session_state.recommender_preloading = True
    threading.Thread(target=preload_recommender, daemon=True).start()


# --- CONFIG ---
st.set_page_config(page_title="Kineto", page_icon="🎬", layout="centered")

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# --- FUNCTIONS ---
# Centralized navigation (solves double-click issue)
def go_to(page):
    st.session_state.page = page
    st.rerun()  # instantly re-run after navigation

# --- RESET WARNING ON PAGE LOAD ---
if "show_warning" not in st.session_state:
    st.session_state.show_warning = False
else:
    # Automatically clear warning when page changes
    if st.session_state.get("last_page") != st.session_state.page:
        st.session_state.show_warning = False
st.session_state.last_page = st.session_state.page

#------------------------------------------------------------------------------------------------------------------
# --- PAGE 1: WELCOME ---
if st.session_state.page == 'welcome':
    st.markdown("""
        <style>
        @keyframes flyLeft { 0%{opacity:0;transform:translateX(-100px)rotate(-20deg);}80%{opacity:1;transform:translateX(10px)rotate(2deg);}100%{opacity:1;transform:translateX(0)rotate(0);} }
        @keyframes flyRight { 0%{opacity:0;transform:translateX(100px)rotate(20deg);}80%{opacity:1;transform:translateX(-10px)rotate(-2deg);}100%{opacity:1;transform:translateX(0)rotate(0);} }
        @keyframes flyTop { 0%{opacity:0;transform:translateY(-100px)rotate(10deg);}80%{opacity:1;transform:translateY(10px)rotate(-2deg);}100%{opacity:1;transform:translateY(0)rotate(0);} }
        .title-container{text-align:center;font-size:48px;font-weight:bold;margin-bottom:10px;}
        .letter{display:inline-block;color:#E50914;opacity:0;}
        .letter:nth-child(1){animation:flyLeft .8s ease forwards .1s;}
        .letter:nth-child(2){animation:flyRight .8s ease forwards .3s;}
        .letter:nth-child(3){animation:flyTop .8s ease forwards .5s;}
        .letter:nth-child(4){animation:flyLeft .8s ease forwards .7s;}
        .letter:nth-child(5){animation:flyRight .8s ease forwards .9s;}
        .letter:nth-child(6){animation:flyTop .8s ease forwards 1.1s;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="title-container">
            🎬 Welcome to
            <span>
                <span class="letter">K</span>
                <span class="letter">i</span>
                <span class="letter">n</span>
                <span class="letter">e</span>
                <span class="letter">t</span>
                <span class="letter">o</span>
            </span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;'>Your personalized movie recommendation assistant powered by hybrid AI.</p>", unsafe_allow_html=True)

    logo_path = "kineto_logo.png"
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"<div style='text-align:center;'><img src='data:image/png;base64,{logo_base64}' width='250' style='margin:20px auto;'/></div>",
        unsafe_allow_html=True
    )

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color:#E50914;color:white;border:none;border-radius:8px;
            padding:10px 24px;font-size:16px;cursor:pointer;display:block;margin:0 auto;
        }
        div.stButton > button:hover{background-color:#b0060f;}
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3.2, 2, 3])
    with col2:
        if st.button("Get Started ➡️"):
            go_to('auth_menu')
#------------------------------------------------------------------------------------------------------------------
# --- PAGE 2: AUTH MENU ---
elif st.session_state.page == 'auth_menu':
    st.markdown("<h2 style='text-align:center;'>Welcome to Kineto</h2>", unsafe_allow_html=True)
    st.write("Choose how you’d like to continue:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Log In", use_container_width=True):
            go_to('login')
    with col2:
        if st.button("🆕 Create Account", use_container_width=True):
            go_to('signup')
    if st.button("⬅ Back"):
        go_to('welcome')
#------------------------------------------------------------------------------------------------------------------
# --- PAGE 3: SIGN UP ---
elif st.session_state.page == 'signup':
    st.markdown("<h2 style='text-align:center;'>Create Your Kineto Account</h2>", unsafe_allow_html=True)
    email = st.text_input("Email")
    password = st.text_input("Create Password", type="password")
    verify = st.text_input("Verify Password", type="password")

    col1, col2, col3 = st.columns([3, 2, 3])
    with col1:
        if st.button("⬅ Back"):
            go_to('auth_menu')

    with col3:
        right_col1, right_col2 = st.columns([1, 1.15])
        with right_col2:
            if st.button("Create Account"):
                # Reset previous messages
                st.session_state.warning_msg = None
                st.session_state.error_msg = None

                if not email or not password or not verify:
                    st.session_state.warning_msg = "⚠️ Please fill out all fields."
                elif password != verify:
                    st.session_state.error_msg = "❌ Passwords do not match."
                else:
                    st.success("Account created successfully! Redirecting...")
                    time.sleep(1.5)
                    go_to('profile')

    # --- Global feedback messages (full width, centered) ---
    if st.session_state.get("warning_msg"):
        st.warning(st.session_state.warning_msg)

    if st.session_state.get("error_msg"):
        st.error(st.session_state.error_msg)
#------------------------------------------------------------------------------------------------------------------
# --- PAGE 4: CREATE PROFILE ---
elif st.session_state.page == 'profile':
    st.header("Let’s personalize your experience")

    # --- Basic info ---
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("First Name", placeholder="John")
    with col2:
        last_name = st.text_input("Last Name", placeholder="Doe")

    # --- Birthday ---
    col1, col2, col3 = st.columns(3)
    with col1:
        day = st.selectbox("Day", list(range(1, 32)), index=0)
    with col2:
        month = st.selectbox("Month", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
    with col3:
        year = st.selectbox("Year", list(range(1920, 2025)), index=100)

    demographics = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])

    # --- Contact info ---
    st.write("### Contact Information")
    col1, col2 = st.columns([1, 3])
    with col1:
        country_code = st.selectbox("Country Code", ["+1", "+31", "+44", "+33", "+49", "+61", "+65", "+81", "+91"])
    with col2:
        phone = st.text_input("Phone Number", placeholder="e.g. 5551234567")

    # --- Address ---
    st.write("### Address Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.text_input("City")
    with col2:
        state = st.selectbox("State / Province", ["Select...", "California", "Florida", "New York", "Texas", "Other"])
    with col3:
        country = st.selectbox("Country", ["United States", "Canada", "Netherlands", "United Kingdom", "Germany", "Other"])

    st.write("Click next to continue to your personalized search.")

    # --- Buttons ---
    col_back, col_space, col_next = st.columns([1, 6, 1])
    with col_back:
        if st.button("⬅ Back"):
            go_to('welcome')

    with col_next:
        if st.button("Next ➡"):
            required_fields = [name, last_name, phone, city]
            if any(f.strip() == "" for f in required_fields):
                st.session_state.show_warning = True
            else:
                st.session_state.show_warning = False  # ✅ reset warning on success
                st.session_state.user_info = {
                    "name": name,
                    "last_name": last_name,
                    "day": day,
                    "month": month,
                    "year": year,
                    "demographics": demographics,
                    "phone": f"{country_code} {phone}",
                    "address": {"city": city, "state": state, "country": country}
                }
                go_to('query')

    # --- Warning (inside same page, not global scope) ---
    if st.session_state.get("show_warning", False):
        st.warning("⚠️ Please complete all required fields before continuing.")
#------------------------------------------------------------------------------------------------------------------
# --- PAGE 5: LOG IN ---
elif st.session_state.page == 'login':
    st.markdown("<h2 style='text-align:center;'>Log In to Kineto</h2>", unsafe_allow_html=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2, col3 = st.columns([3, 2, 3])
    with col1:
        if st.button("⬅ Back"):
            go_to('auth_menu')
    with col3:
        right_col1, right_col2 = st.columns([1, 0.45])
        with right_col2:
            if st.button("Log In"):
                if email and password:
                    st.session_state.warning_msg = None # reset previous warnings
                    st.success("Login successful! Redirecting...")
                    time.sleep(1.5)
                    go_to('query')
                else:
                    st.warning("Please enter both email and password.")
#------------------------------------------------------------------------------------------------------------------
# --- PAGE 6: QUERY + RECOMMENDATION ---
elif st.session_state.page == 'query':
    user_info = st.session_state.get("user_info", {})
    name = user_info.get("name", "")

    st.markdown(f"<h2 style='text-align:center;'>Hi {name}! 👋 What do you feel like watching?</h2>", unsafe_allow_html=True)
    st.write("Tell us what you want to watch like you're talking to a friend. Be as descriptive as you'd like!")
    query = st.text_input("(Example: A heartfelt drama about friendship and loyalty)")

    col1, col2, col3 = st.columns([1.5, 5, 3])
    with col1:
        if st.button("⬅ Back"):
            go_to('profile')
    with col3:
        get_recs = st.button("🎥 Get Recommendations", use_container_width=True)

    if get_recs:
        if not query.strip():
            st.warning("Please enter a movie description first.")
        else:
            with st.spinner("Finding the best matches..."):
                recommender = get_cached_recommender()
                results = recommender.recommend(query, user_id=None, n=5)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🎬 Top Recommendations")

            if not results:
                st.error("No recommendations found. Try describing it differently!")
            else:
                for movie in results:
                    st.markdown(
                        f"**{movie['title']}** ({movie.get('year', 'N/A')}) — ⭐ {movie['rating']:.1f} "
                        f"<br><small><i>Themes:</i> {', '.join(movie.get('themes', [])[:3])}</small><br>"
                        f"<small><i>Genres:</i> {', '.join(movie.get('genres', [])[:3])}</small>",
                        unsafe_allow_html=True
                    )

## venv312\Scripts\activate
## pip install -r requirements.txt (for all required libraries)
## streamlit run app.py


## Command prompt: python -m streamlit run app.py
