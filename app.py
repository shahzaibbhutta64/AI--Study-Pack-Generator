import streamlit as st
import json
import os
from workflow import UserInput, StudyPackGenerator

st.set_page_config(page_title="AI Study Pack Generator", page_icon="📚", layout="wide")

st.title("🎓 AI Study Pack Generator")
st.caption("Powered by Groq Hardware Acceleration")

# Retrieve API Key from Streamlit Secrets or Sidebar Input
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar or configure GROQ_API_KEY in Streamlit Secrets to proceed.")
        st.stop()

with st.sidebar:
    st.header("📋 Configuration & Inputs")
    subject = st.text_input("Subject / Topic", "Python Data Structures")
    objective = st.text_input("Learning Objective", "Master Lists, Dicts, and Sets")
    level = st.selectbox("Knowledge Level", ["Beginner", "Intermediate", "Advanced"], index=1)
    duration = st.text_input("Study Duration", "2 Hours")
    style = st.text_input("Learning Style", "Visual / Code Examples")
    difficulty = st.selectbox("Desired Difficulty", ["Easy", "Medium", "Hard"], index=1)
    goal = st.text_input("Exam or Goal", "Technical Interview")
    num_q = st.number_input("Number of Practice Questions", min_value=1, max_value=10, value=3)
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
        with st.spinner("Processing multi-stage AI workflow..."):
            final_state = generator.run_pipeline(user_input, progress_cb=update_progress)
        
        status_text.success("✅ Study Pack Successfully Generated!")
        
        tab1, tab2, tab3 = st.tabs(["📖 Study Notes", "📝 Assessment", "🔍 Audit & Raw JSON"])
        
        # TAB 1: Notes & Tips
        with tab1:
            st.markdown("## 📖 Study Notes")
            refinement_data = final_state.refinement if isinstance(final_state.refinement, dict) else {}
            
            notes = refinement_data.get("refined_notes", "No notes generated.")
            tips = refinement_data.get("exam_tips", [])
            
            st.markdown(notes)
            
            st.markdown("---")
            st.markdown("### 💡 Exam & Study Tips")
            if isinstance(tips, list) and tips:
                for tip in tips:
                    st.write(f"- {tip}")
            elif isinstance(tips, str):
                st.write(tips)
            else:
                st.info("No explicit exam tips generated.")
                
        # TAB 2: Questions & Answers
        with tab2:
            st.markdown("## 📝 Practice Assessment")
            refinement_data = final_state.refinement if isinstance(final_state.refinement, dict) else {}
            assessment_data = refinement_data.get("refined_assessment", {})
            
            if not isinstance(assessment_data, dict):
                assessment_data = final_state.assessment if isinstance(final_state.assessment, dict) else {}
                
            questions = assessment_data.get("questions", []) if isinstance(assessment_data, dict) else []
            
            if isinstance(questions, list) and questions:
                for idx, q in enumerate(questions):
                    if isinstance(q, dict):
                        q_id = q.get('id', idx + 1)
                        q_text = q.get('question', 'Question text unavailable')
                        q_diff = q.get('difficulty', '')
                        
                        with st.expander(f"Question {q_id}: {q_text} ({q_diff})"):
                            options = q.get("options", [])
                            if isinstance(options, list) and options:
                                st.write("**Options:**")
                                for opt in options:
                                    st.write(f"- {opt}")
                            st.success(f"**Correct Answer:** {q.get('answer', 'N/A')}")
                            st.info(f"**Explanation:** {q.get('explanation', 'No explanation provided.')}")
            else:
                st.warning("No structured questions could be parsed from the model output.")

        # TAB 3: Debug / Raw Data
        with tab3:
            st.markdown("## 🔍 Workflow Execution Data")
            st.json(final_state.model_dump())

    except Exception as e:
        st.error(f"❌ Error during execution: {str(e)}")
