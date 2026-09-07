import json
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel
from groq import Groq

class UserInput(BaseModel):
    subject: str
    learning_objective: str
    knowledge_level: str
    study_duration: str
    preferred_learning_style: str
    desired_difficulty: str
    exam_goal: str
    num_questions: int
    additional_context: Optional[str] = ""

class WorkflowState(BaseModel):
    inputs: UserInput
    plan: Optional[Dict[str, Any]] = None
    content: Optional[Dict[str, Any]] = None
    assessment: Optional[Dict[str, Any]] = None
    review: Optional[Dict[str, Any]] = None
    refinement: Optional[Dict[str, Any]] = None

class StudyPackGenerator:
    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-120b"):
        self.client = Groq(api_key=api_key)
        self.model = model_name

    def _call_groq_json(self, system_prompt: str, user_prompt: str, retries: int = 2) -> Dict[str, Any]:
        """Calls Groq API enforcing JSON output with retry logic and fallback parsing."""
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt + "\nIMPORTANT: You MUST return strictly a valid JSON object."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                raw_content = response.choices[0].message.content
                parsed = json.loads(raw_content)
                if isinstance(parsed, dict):
                    return parsed
                return {"result": str(parsed)}
            except Exception as e:
                if attempt == retries:
                    raise RuntimeError(f"API stage execution failed: {str(e)}")
                time.sleep(1)

    def stage_1_planning(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are an expert curriculum designer."
        user_prompt = f"""Create a structured study plan based on user inputs:
{state.inputs.model_dump_json()}

Return JSON format with keys: "topics", "learning_objectives", "difficulty_progression", "recommended_sequence"."""
        state.plan = self._call_groq_json(system_prompt, user_prompt)
        return state

    def stage_2_content(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are an educational author."
        plan_data = state.plan if isinstance(state.plan, dict) else {}
        user_prompt = f"""Generate study material matching this plan:
{json.dumps(plan_data)}

Return JSON format with keys: "topic_explanations", "key_concepts", "definitions", "examples", "memory_aids"."""
        state.content = self._call_groq_json(system_prompt, user_prompt)
        return state

    def stage_3_assessment(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are an assessment writer."
        content_data = state.content if isinstance(state.content, dict) else {}
        user_prompt = f"""Generate exactly {state.inputs.num_questions} questions based strictly on this content:
{json.dumps(content_data)}

Return JSON format with key "questions" containing a list of objects with keys: "id", "question", "options", "answer", "explanation", "difficulty"."""
        state.assessment = self._call_groq_json(system_prompt, user_prompt)
        return state

    def stage_4_review(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are an educational auditor."
        p_data = state.plan if isinstance(state.plan, dict) else {}
        c_data = state.content if isinstance(state.content, dict) else {}
        a_data = state.assessment if isinstance(state.assessment, dict) else {}
        
        user_prompt = f"""Review the generated material for clarity, consistency, and accuracy:
Plan: {json.dumps(p_data)}
Content: {json.dumps(c_data)}
Assessment: {json.dumps(a_data)}

Return JSON format with keys: "is_valid", "errors_found", "quality_score", "improvement_feedback"."""
        state.review = self._call_groq_json(system_prompt, user_prompt)
        return state

    def stage_5_refinement(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are an expert editor."
        r_data = state.review if isinstance(state.review, dict) else {}
        c_data = state.content if isinstance(state.content, dict) else {}
        a_data = state.assessment if isinstance(state.assessment, dict) else {}

        user_prompt = f"""Refine the study pack using review feedback:
Feedback: {json.dumps(r_data)}
Content: {json.dumps(c_data)}
Assessment: {json.dumps(a_data)}

Return JSON format with keys: "refined_notes", "refined_assessment", "exam_tips"."""
        
        res = self._call_groq_json(system_prompt, user_prompt)
        
        # Guard clause: ensure state.refinement is guaranteed to be a dictionary
        if isinstance(res, dict):
            state.refinement = res
        else:
            state.refinement = {
                "refined_notes": str(res),
                "refined_assessment": a_data,
                "exam_tips": []
            }
        return state

    def run_pipeline(self, inputs: UserInput, progress_cb=None) -> WorkflowState:
        state = WorkflowState(inputs=inputs)
        stages = [
            ("Stage 1: Planning", self.stage_1_planning),
            ("Stage 2: Content Generation", self.stage_2_content),
            ("Stage 3: Assessment Creation", self.stage_3_assessment),
            ("Stage 4: Quality Audit", self.stage_4_review),
            ("Stage 5: Refinement & Polishing", self.stage_5_refinement)
        ]
        for idx, (name, fn) in enumerate(stages):
            if progress_cb:
                progress_cb(f"Processing {name}...", (idx + 1) / len(stages))
            state = fn(state)
        return state
