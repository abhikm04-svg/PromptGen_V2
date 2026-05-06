"""AI Agents for PromptGen V2 - using google-genai SDK"""
import re
from typing import Dict, List, Tuple, Optional, Callable
from google import genai
from google.genai import types
from core.config import (
    PROMPT_GENERATOR_MODEL, PROMPT_TESTER_MODEL, ANALYZER_MODEL, get_client
)

# ============================================================================
# System Prompts
# ============================================================================

QUESTION_GENERATOR_SYSTEM = """You are a Question Generator AI agent. Your role is to help users define their prompt requirements by asking comprehensive, targeted questions.

When a user shares their idea, ask 3-5 relevant questions to gather essential information:

1. **Purpose & Goal**: What is the main objective or desired outcome?
2. **Target Audience**: Who will be using or reading this content?
3. **Tone & Style**: What tone should the output have? (formal, casual, technical, creative, etc.)
4. **Context & Constraints**: Are there specific requirements, limitations, or guidelines to follow?
5. **Output Format**: What format should the final output be in? (essay, bullet points, code, summary, etc.)
6. **Key Elements**: What specific elements or information must be included?
7. **Examples**: Are there any examples or references to follow?

Ask these questions in a clear, conversational manner. Wait for the user's responses before proceeding. Your output should be ONLY the questions - do not create the prompt yet. Number each question clearly."""

PROMPT_GENERATOR_SYSTEM = """You are an expert prompt engineer. Based on the user's answers to the questions, create a comprehensive and well-structured prompt that addresses all the requirements. The prompt should be clear, specific, and optimized for achieving the user's goals. Include all relevant details about purpose, audience, tone, constraints, and desired output format. Return ONLY the created prompt text without any additional explanation. The prompt must be XML tagged. Below are the tags that you can use:
<role> (Mandatory Tag)
<context> (Mandatory Tag)
<instructions> (Mandatory Tag)
<skills> (Mandatory Tag)
<areas of knowledge> (Mandatory Tag)
<next steps> (Mandatory Tag)
<personality and style> (Mandatory Tag)
<output format> (Mandatory Tag)
<additional instructions> (Mandatory Tag) (To be filled by the user)
<audience> (Optional)
<examples> (Optional)
<dont's> (Optional)
Ensure that the final prompt output when the user asks the prompt to the AI model should be in the same language as the user's idea and should be in user readable format and non xml tagged format."""

ANALYZER_PROMPT_TEMPLATE = """You will take the prompt [{prompt}] and benchmark it extremely strictly against the following parameters:

**Prompt Benchmark Attributes:**

1. **Clarity & Unambiguity**: How easily understandable is the prompt's language and intent?
   - 1: Very ambiguous or confusing language.
   - 5: Mostly clear, but requires some interpretation.
   - 10: Perfectly clear, intent is obvious.

2. **Specificity & Detail**: How precisely does the prompt define the desired outcome and its characteristics?
   - 1: Very vague, lacks necessary details.
   - 5: Moderately specific, includes some key details.
   - 10: Highly specific, provides comprehensive details about the desired output.

3. **Contextual Sufficiency**: Does the prompt provide all necessary background information for the AI to understand the task domain?
   - 1: Lacks critical context, making the task difficult to perform accurately.
   - 5: Provides some necessary context, but gaps remain.
   - 10: Provides all relevant context required for the task.

4. **Constraint Definition**: How clearly are limitations (e.g., length, format, tone, exclusions, required elements) stated?
   - 1: No constraints defined, or constraints are unclear.
   - 5: Some relevant constraints are mentioned but may be incomplete or slightly ambiguous.
   - 10: All relevant constraints are clearly and explicitly defined.

5. **Role/Persona Clarity**: If a specific role or persona for the AI is requested, how clearly is it defined? (Score NA if no role is needed).
   - 1: Role is undefined or very confusing.
   - 5: Role is mentioned but lacks detail or nuance.
   - 10: Role is clearly defined with sufficient detail to guide the AI's perspective.

6. **Task Definition**: How clearly is the primary goal or task articulated? What should the AI do?
   - 1: The core task is unclear or poorly defined.
   - 5: The task is generally understandable but could be more precise.
   - 10: The task is explicitly and accurately defined.

7. **Conciseness & Efficiency**: Is the prompt free from irrelevant information, redundancy, or excessive length?
   - 1: Very verbose, contains significant irrelevant information.
   - 5: Mostly concise, with minor irrelevant parts.
   - 10: Optimally concise, containing only necessary information.

8. **Structural Organization**: Is the prompt logically structured (e.g., using formatting, clear separation of instructions)?
   - 1: Disorganized, hard to follow instructions.
   - 5: Moderately organized, structure could be improved.
   - 10: Well-structured, logical flow, easy to parse.

9. **Instructional Effectiveness**: How likely are the specific instructions and phrasing to lead directly to the desired output, minimizing potential misinterpretation?
   - 1: Instructions are likely to be misinterpreted or fail.
   - 5: Instructions are somewhat effective but could lead to deviations.
   - 10: Instructions are highly effective and precisely guide the AI.

10. **Tone/Style Guidance**: How clearly is the desired tone (e.g., formal, friendly, technical) or writing style (e.g., simple, academic, bullet points) specified? (Score NA if not relevant).
    - 1: No guidance, or guidance is vague/contradictory.
    - 5: Some guidance provided, but could be more specific.
    - 10: Clear, specific, and consistent guidance on tone/style.

Next you will take the prompt output [{output}] and benchmark it extremely strictly against the following parameters:

**Output Benchmark Attributes:**

1. **Relevance to Prompt**: How directly does the output address the core request and topic of the prompt?
   - 1: Off-topic or completely misses the prompt's intent.
   - 5: Partially relevant, addresses some aspects but deviates or includes irrelevant info.
   - 10: Directly and fully relevant to the prompt's core request.

2. **Factual Accuracy**: How correct is the factual information presented in the output? (Score NA if subjective/creative).
   - 1: Contains significant factual errors or hallucinations.
   - 5: Mostly accurate, but contains minor inaccuracies.
   - 10: Completely factually accurate.

3. **Completeness of Response**: Does the output address all explicit and implicit requirements, questions, or parts of the prompt?
   - 1: Misses major requirements or questions asked.
   - 5: Addresses most requirements but omits minor aspects.
   - 10: Fully addresses all requirements specified in the prompt.

4. **Coherence & Logical Flow**: Is the output well-organized, with logical connections between ideas/sentences/paragraphs?
   - 1: Incoherent, disjointed, very difficult to follow.
   - 5: Mostly coherent, but some transitions or points are unclear.
   - 10: Very coherent, logical flow, easy to follow.

5. **Clarity & Readability**: Is the language used clear, grammatically correct, and easy for the target audience to understand?
   - 1: Very difficult to understand, poor grammar, confusing language.
   - 5: Mostly clear, but some awkward phrasing or minor grammatical errors.
   - 10: Very clear, well-written, and easily readable.

6. **Constraint Adherence**: Does the output respect all constraints (length, format, style, exclusions) specified in the prompt?
   - 1: Ignores most or all specified constraints.
   - 5: Meets some constraints but violates others.
   - 10: Meets all specified constraints perfectly.

7. **Tone & Style Consistency**: Does the output consistently match the tone and style requested (or implied) by the prompt?
   - 1: Uses a completely inappropriate tone/style or is highly inconsistent.
   - 5: Mostly matches the requested tone/style but has inconsistencies.
   - 10: Perfectly and consistently matches the requested tone/style.

8. **Formatting & Presentation**: Is the output presented in a clean, usable, and readable format (using markdown, lists, code blocks, etc., appropriately)?
   - 1: Poor or absent formatting, difficult to read/use.
   - 5: Adequate formatting, generally readable.
   - 10: Excellent formatting and presentation enhances readability and usability.

9. **Depth & Elaboration**: Does the output provide sufficient detail, explanation, or depth appropriate for the prompt's request? (Consider if the prompt asked for brevity vs. detail).
   - 1: Too superficial, lacks necessary detail or explanation.
   - 5: Provides adequate depth for the request.
   - 10: Provides comprehensive and insightful detail, exceeding expectations where appropriate.

10. **Helpfulness & Actionability**: How well does the output actually achieve the underlying goal of the prompt? Is it useful, practical, or actionable?
    - 1: Not helpful, doesn't achieve the prompt's goal.
    - 5: Somewhat helpful, partially achieves the goal.
    - 10: Very helpful, effectively achieves the prompt's goal, potentially offering extra value.

Based on your evaluation, rate the prompt's effectiveness on a scale of 0 to 100 with 100 being perfect and 0 being completely unusable. Be extremely strict during benchmarking - you should ensure that the prompt is absolutely perfect.

**Output Format:**
Score: [0-100]
Feedback: [Points of improvement for the prompt]

Only output the score and points of improvement as feedback. No additional information is required."""


# ============================================================================
# Helper: generate content with streaming support
# ============================================================================

def _generate(model_name: str, prompt: str, temperature: float = None,
              stream_callback: Optional[Callable] = None, agent_tag: str = "") -> str:
    """Generate content using the google-genai Client."""
    client = get_client()
    if client is None:
        raise RuntimeError("API client not configured. Call configure_client() first.")

    config = {}
    if temperature is not None:
        config["temperature"] = temperature

    gen_config = types.GenerateContentConfig(**config) if config else None

    full_text = ""
    try:
        if stream_callback:
            response = client.models.generate_content_stream(
                model=model_name, contents=prompt, config=gen_config
            )
            for chunk in response:
                try:
                    if chunk.text:
                        full_text += chunk.text
                        stream_callback(chunk.text, agent_tag)
                except (ValueError, AttributeError):
                    continue
            return full_text.strip()
        else:
            response = client.models.generate_content(
                model=model_name, contents=prompt, config=gen_config
            )
            return response.text.strip()
    except Exception as e:
        print(f"Error in _generate ({agent_tag}): {e}")
        return ""


# ============================================================================
# Question Generator Agent
# ============================================================================

class QuestionGeneratorAgent:
    """Generates 3-5 clarification questions based on user's prompt idea"""

    def __init__(self, model_name: str = PROMPT_GENERATOR_MODEL):
        self.model_name = model_name

    def generate_questions(self, user_idea: str, stream_callback: Optional[Callable] = None) -> List[str]:
        prompt = f"""{QUESTION_GENERATOR_SYSTEM}

User's idea: {user_idea}

Please ask 3-5 relevant questions to clarify the requirements."""

        text = _generate(self.model_name, prompt, stream_callback=stream_callback,
                         agent_tag="question_generator")

        if not text:
            return [
                "What is the main purpose or goal of this prompt?",
                "Who is the target audience for the output?",
                "What tone or style should the output have?"
            ]
        return self._parse_questions(text)

    def _parse_questions(self, text: str) -> List[str]:
        questions = []
        pattern = r'\d+[.)]\s*(.+?)(?=\d+[.)]|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            questions = [q.strip() for q in matches]
        else:
            parts = re.split(r'\?+', text)
            questions = [q.strip() + '?' for q in parts if q.strip() and '?' not in q]
        questions = [q for q in questions if len(q) > 10]
        if len(questions) < 3:
            questions = [q.strip() for q in text.split('\n') if '?' in q and len(q.strip()) > 10][:5]
        return questions[:5]


# ============================================================================
# Prompt Generator Agent
# ============================================================================

class PromptGeneratorAgent:
    """Generates optimized prompts based on user idea and clarification answers"""

    def __init__(self, model_name: str = PROMPT_GENERATOR_MODEL):
        self.model_name = model_name

    def generate_prompt(self, user_idea: str, user_answers: Dict[str, str],
                        feedback: str = "", stream_callback: Optional[Callable] = None) -> str:
        answers_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in user_answers.items()])

        if feedback:
            prompt = f"""{PROMPT_GENERATOR_SYSTEM}

Previous prompt feedback: {feedback}

User's original idea: {user_idea}

User's clarification answers:
{answers_text}

Based on the feedback, improve the prompt to address the issues mentioned. Return ONLY the improved prompt with XML tags."""
        else:
            prompt = f"""{PROMPT_GENERATOR_SYSTEM}

User's original idea: {user_idea}

User's clarification answers:
{answers_text}

Create a comprehensive prompt based on the above information. Return ONLY the prompt with XML tags."""

        return _generate(self.model_name, prompt, temperature=0.1,
                         stream_callback=stream_callback, agent_tag="prompt_generator")


# ============================================================================
# Prompt Tester Agent
# ============================================================================

class PromptTesterAgent:
    """Tests the generated prompt by running it on the model"""

    def __init__(self, model_name: str = PROMPT_TESTER_MODEL):
        self.model_name = model_name

    def test_prompt(self, prompt: str, stream_callback: Optional[Callable] = None) -> str:
        result = _generate(self.model_name, prompt, temperature=0.2,
                           stream_callback=stream_callback, agent_tag="prompt_tester")
        return result if result else "Error: Failed to generate output"


# ============================================================================
# Prompt Analyzer Agent
# ============================================================================

class PromptAnalyzerAgent:
    """Analyzes and scores prompts based on multiple parameters"""

    def __init__(self, model_name: str = ANALYZER_MODEL):
        self.model_name = model_name

    def analyze(self, prompt: str, output: str, stream_callback: Optional[Callable] = None) -> Tuple[int, str]:
        analysis_prompt = ANALYZER_PROMPT_TEMPLATE.format(prompt=prompt, output=output)
        text = _generate(self.model_name, analysis_prompt, temperature=0.2,
                         stream_callback=stream_callback, agent_tag="analyzer")
        if not text:
            return 0, "Error during analysis"
        return self._parse_response(text)

    def _parse_response(self, response_text: str) -> Tuple[int, str]:
        score_match = re.search(r'Score:\s*(\d+)', response_text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
        else:
            score_match = re.search(r'(\d+)\s*/\s*100', response_text)
            score = int(score_match.group(1)) if score_match else 0

        feedback_match = re.search(r'Feedback:\s*(.+?)(?=\n\n|$)', response_text, re.DOTALL | re.IGNORECASE)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        else:
            parts = re.split(r'Score:\s*\d+', response_text, flags=re.IGNORECASE)
            feedback = parts[1].strip() if len(parts) > 1 else "No feedback provided."

        return score, feedback
