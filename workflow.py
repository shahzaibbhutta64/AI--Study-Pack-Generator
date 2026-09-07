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
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt == retries:
                    raise RuntimeError(f"Error in stage: {str(e)}")
                time.sleep(1)

    def stage_1_planning(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are an educational planner."
        user_prompt = f"Create plan: {state.inputs.model_dump_json()}\nReturn JSON keys: topics, learning_objectives, difficulty_progression, recommended_sequence."
        state.plan = self._call_groq_json(system_prompt, user_prompt)
        return state

    def stage_2_content(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are a content writer."
        user_prompt = f"Create notes for plan: {json.dumps(state.plan)}\nReturn JSON keys: topic_explanations, key_concepts, definitions, examples, memory_aids."
        state.content = self._call_groq_json(system_prompt, user_prompt)
        return state

    def stage_3_assessment(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are an examiner."
        user_prompt = f"Create {state.inputs.num_questions} questions for content: {json.dumps(state.content)}\nReturn JSON key 'questions' with question list."
        state.assessment = self._call_groq_json(system_prompt, user_prompt)
        return state

    def stage_4_review(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are a quality reviewer."
        user_prompt = f"Audit quality: Plan {json.dumps(state.plan)}, Content {json.dumps(state.content)}, Assessment {json.dumps(state.assessment)}\nReturn JSON keys: is_valid, errors_found, quality_score, improvement_feedback."
        state.review = self._call_groq_json(system_prompt, user_prompt)
        return state

    def stage_5_refinement(self, state: WorkflowState) -> WorkflowState:
        system_prompt = "You are an editor."
        user_prompt = f"Refine pack using review feedback: {json.dumps(state.review)}\nContent: {json.dumps(state.content)}\nAssessment: {json.dumps(state.assessment)}\nReturn JSON keys: refined_notes, refined_assessment, exam_tips."
        state.refinement = self._call_groq_json(system_prompt, user_prompt)
        return state

    def run_pipeline(self, inputs: UserInput, progress_cb=None) -> WorkflowState:
        state = WorkflowState(inputs=inputs)
        stages = [
            ("Planning", self.stage_1_planning),
            ("Content Generation", self.stage_2_content),
            ("Assessment Creation", self.stage_3_assessment),
            ("Quality Review", self.stage_4_review),
            ("Refinement", self.stage_5_refinement)
        ]
        for idx, (name, fn) in enumerate(stages):
            if progress_cb:
                progress_cb(f"Step {idx+1}/5: {name}...", (idx + 1) / len(stages))
            state = fn(state)
        return state
