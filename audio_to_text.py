def recognize_speech_from_mic():
    """Listens for speech via microphone and transcribes it using Google Web Speech API."""
    # Check for PyAudio installation
    try:
        import pyaudio
    except Exception as e:
        st.error("PyAudio library is not installed. Please install it to use the microphone: `pip install PyAudio`")
        return ""
 
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.info("Listening... Please speak your complaint.")
 
        try:
            # Listen for speech with a timeout and phrase time limit
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=20)
        except sr.WaitTimeoutError:
            st.warning("Listening timed out. No speech was detected.")
            return ""
 
    with st.spinner("Recognizing speech..."):
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
 
# --- Streamlit UI ---
def render_submission_form():
    """Renders the initial complaint submission form."""
    st.markdown(
        "Submit your complaint by typing in the text box or by recording your voice directly."
    )
 
    # Use a form to group inputs and the submission button
    with st.form(key="complaint_form"):
        complaint_text_input = st.text_area(
            "Enter your complaint text here, or use the button below to record.",
            value=st.session_state.get("complaint_text", ""),
            height=150,
            key="complaint_text_area"
        )
 
        # Added email input
        complainer_email_input = st.text_input(
            "Enter your email to receive updates (optional)",
            key="complainer_email_input"
        )
 
        image_file = st.file_uploader(
            "Upload an image (optional)", type=["jpg", "jpeg", "png"]
        )
 
        submitted = st.form_submit_button("Submit Complaint")
 
    if st.button("🎤 Record Complaint via Microphone"):
        transcribed_text = recognize_speech_from_mic()
        if transcribed_text:
            st.session_state.complaint_text = transcribed_text
            st.rerun()
