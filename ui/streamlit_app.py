import requests
import streamlit as st


# where the FastAPI backend runs INSIDE the container (Streamlit talks to it here)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Dialogue Summarizer", page_icon="📝")
st.title("📝 Dialogue Summarizer")
st.caption("Paste a conversation and get a short summary — from a Pegasus model I fine-tuned on the SAMSum dataset.")

# track whether we've already had a successful summary this session, so the
# "loading the model" note only shows when it's actually likely to be loading
if "warmed_up" not in st.session_state:
    st.session_state.warmed_up = False

# the box where I paste the dialogue
dialogue = st.text_area(
    "Conversation",
    height=220,
    placeholder="Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check.\nAmanda: Sorry, can't find it.\nHannah: Ok, thanks.",
)

if st.button("Summarize", type="primary"):
    if not dialogue.strip():
        st.warning("Paste a conversation first.")
    else:
        # only warn about the slow model load if we haven't warmed up yet this session
        if st.session_state.warmed_up:
            spinner_msg = "Summarizing..."
        else:
            spinner_msg = "Summarizing... (first run may load the model, so it's slower)"

        with st.spinner(spinner_msg):
            try:
                # /predict takes `text` as a query param, so I send it with params=
                resp = requests.post(f"{API_URL}/predict", params={"text": dialogue}, timeout=180)
                if resp.status_code == 200:
                    # the endpoint returns the summary as a JSON string; fall back to raw text if needed
                    try:
                        summary = resp.json()
                    except Exception:
                        summary = resp.text
                    st.session_state.warmed_up = True   # model is loaded now, don't warn next time
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Could not reach the API: {e}")