import streamlit as st
import json
import os
from workflow import UserInput, StudyPackGenerator

st.set_page_config(page_title="AI Study Pack Generator", page_icon="📚", layout="wide")

st.title("🎓 AI Study Pack Generator")
st.caption("Powered by Groq Acceleration")

groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")
    if not groq_api_key:
        st.warning("⚠️ Please provide a Groq API key to proceed.")
        st.stop()

with st.sidebar:
    st.header("📋 Input Details")
    subject = st.text_input("Subject", "Data Structures")
    objective = st.text_input("Learning Objective", "Understand Trees & Graphs")
    level = st.selectbox("Knowledge Level", ["Beginner", "Intermediate", "Advanced"], index=1)
    duration = st.text_input("Study Duration", "2 Hours")
    style = st.text_input("Learning Style", "Visual examples")
    difficulty = st.selectbox("Desired Difficulty", ["Easy", "Medium", "Hard"], index=1)
    goal = st.text_input("Exam or Goal", "Midterm Exam")
    num_q = st.number_input("Number of Questions", min_value=1, max_value=10, value=3)
    context = st.text_area("Additional Context", "Focus on practical time complexity")
    
    generate_btn = st.button("🚀 Generate Study Pack", type="primary")

if generate_btn:
    user_input = UserInput(
        subject=subject,
        learning_objective=objective,
        knowledge_level=level,
        study_duration=duration,
        preferred_learning_style=style,
        desired_difficulty=difficulty,
        exam_goal=goal,
        num_questions=int(num_q),
        additional_context=context
    )
    
    generator = StudyPackGenerator(api_key=groq_api_key)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(msg, pct):
        status_text.text(msg)
        progress_bar.progress(pct)

    try:
        with st.spinner("Processing through 5 workflow stages..."):
            final_state = generator.run_pipeline(user_input, progress_cb=update_progress)
        
        status_text.success("✅ Study Pack Successfully Generated!")
        
        tab1, tab2, tab3 = st.tabs(["📖 Study Notes", "📝 Questions", "🔍 Audit & Raw JSON"])
        
        with tab1:
            st.markdown(final_state.refinement.get("refined_notes", ""))
            st.markdown("### 💡 Exam Tips")
            for tip in final_state.refinement.get("exam_tips", []):
                st.write(f"- {tip}")
                
        with tab2:
            questions = final_state.refinement.get("refined_assessment", {}).get("questions", [])
            for q in questions:
                with st.expander(f"Q{q.get('id')}: {q.get('question')}"):
                    if "options" in q:
                        for opt in q["options"]:
                            st.write(f"- {opt}")
                    st.success(f"**Answer:** {q.get('answer')}")
                    st.info(f"**Explanation:** {q.get('explanation')}")

        with tab3:
            st.json(final_state.model_dump())

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
