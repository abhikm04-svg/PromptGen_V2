"""
Prompt Optimizer Workflow

This module implements a multi-agent system for generating and optimizing prompts using Google Gemini models.

Workflow:
1. User enters an idea for a prompt
2. Question Generator Agent asks 3-5 clarification questions
3. User answers the questions
4. Prompt Generator Agent creates the prompt (using gemini-2.5-flash)
5. Prompt Tester Agent tests the prompt (using gemini-2.5-pro)
6. Prompt Analyzer Agent scores the prompt and output (0-100)
7. If score < 100, loop back to improve (max 15 iterations)
8. Display final prompt to user
"""

# ============================================================================
# 1. Import Libraries and Setup
# ============================================================================

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "google-generativeai package is not installed. "
        "Please install it using: pip install google-generativeai"
    )

import os
import json
import re
import time
import html
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
try:
    import pandas as pd
except ImportError:
    pd = None  # pandas will be available in Streamlit environment

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import io
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    io = None

# ============================================================================
# 2. Configuration
# ============================================================================

def get_api_key():
    """Get API key from environment variable (for non-Streamlit usage)"""
    return os.getenv('GOOGLE_API_KEY', None)

# Don't configure API key at module level for Streamlit apps
# It will be configured in main() function from Streamlit secrets
# For non-Streamlit usage, configure from environment variable
GOOGLE_API_KEY = get_api_key()
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Model configurations
# Note: Model names may need to be adjusted based on available models in the API
# Common alternatives: 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp'
PROMPT_GENERATOR_MODEL = 'gemini-2.5-flash'  # Using flash for prompt generation
PROMPT_TESTER_MODEL = 'gemini-2.5-pro'  # Using pro for testing
ANALYZER_MODEL = 'gemini-2.5-pro'  # Using pro for analysis

# Workflow settings
MAX_ITERATIONS = 15
TARGET_SCORE = 100

# ============================================================================
# 3. Data Classes for State Management
# ============================================================================

@dataclass
class WorkflowState:
    """Tracks the state of the workflow"""
    user_idea: str = ""
    clarification_questions: List[str] = None
    user_answers: Dict[str, str] = None
    current_prompt: str = ""
    prompt_output: str = ""
    current_score: int = 0
    feedback: str = ""
    iteration: int = 0
    best_prompt: str = ""
    best_score: int = 0
    best_output: str = ""
    token_usage: List[Dict] = field(default_factory=list)  # Track token usage over time
    
    def __post_init__(self):
        if self.clarification_questions is None:
            self.clarification_questions = []
        if self.user_answers is None:
            self.user_answers = {}
    
    def update_best(self):
        """Update best prompt if current score is higher"""
        if self.current_score > self.best_score:
            self.best_score = self.current_score
            self.best_prompt = self.current_prompt
            self.best_output = self.prompt_output
    
    def add_token_usage(self, timestamp: float, tokens: int, stage: str):
        """Add token usage data point"""
        self.token_usage.append({
            'timestamp': timestamp,
            'tokens': tokens,
            'stage': stage,
            'cumulative': sum(t['tokens'] for t in self.token_usage) + tokens
        })

# ============================================================================
# 4. Question Generator Agent
# ============================================================================

class QuestionGeneratorAgent:
    """Generates 3-5 clarification questions based on user's prompt idea"""
    
    def __init__(self, model_name: str = PROMPT_GENERATOR_MODEL):
        self.model = genai.GenerativeModel(model_name)
        self.system_prompt = """You are a Question Generator AI agent. Your role is to help users define their prompt requirements by asking comprehensive, targeted questions.

When a user shares their idea, ask 3-5 relevant questions to gather essential information:

1. **Purpose & Goal**: What is the main objective or desired outcome?
2. **Target Audience**: Who will be using or reading this content?
3. **Tone & Style**: What tone should the output have? (formal, casual, technical, creative, etc.)
4. **Context & Constraints**: Are there specific requirements, limitations, or guidelines to follow?
5. **Output Format**: What format should the final output be in? (essay, bullet points, code, summary, etc.)
6. **Key Elements**: What specific elements or information must be included?
7. **Examples**: Are there any examples or references to follow?

Ask these questions in a clear, conversational manner. Wait for the user's responses before proceeding. Your output should be ONLY the questions - do not create the prompt yet. Number each question clearly."""
    
    def generate_questions(self, user_idea: str, stream_callback: Optional[Callable] = None) -> List[str]:
        """Generate clarification questions based on user's idea"""
        prompt = f"""{self.system_prompt}

User's idea: {user_idea}

Please ask 3-5 relevant questions to clarify the requirements."""
        
        try:
            full_text = ""
            if stream_callback:
                # Stream the response
                response = self.model.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        full_text += chunk.text
                        stream_callback(chunk.text, "question_generator")
                questions_text = full_text.strip()
            else:
                response = self.model.generate_content(prompt)
                questions_text = response.text.strip()
            
            # Parse questions (numbered list)
            questions = self._parse_questions(questions_text)
            return questions
        except Exception as e:
            print(f"Error generating questions: {e}")
            # Fallback questions
            return [
                "What is the main purpose or goal of this prompt?",
                "Who is the target audience for the output?",
                "What tone or style should the output have?"
            ]
    
    def _parse_questions(self, text: str) -> List[str]:
        """Parse questions from the response text"""
        # Split by numbered patterns (1., 2., etc.) or question marks
        questions = []
        # Try to split by numbered list
        pattern = r'\d+[.)]\s*(.+?)(?=\d+[.)]|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            questions = [q.strip() for q in matches]
        else:
            # Fallback: split by question marks
            parts = re.split(r'\?+', text)
            questions = [q.strip() + '?' for q in parts if q.strip() and '?' not in q]
        
        # Clean up questions
        questions = [q for q in questions if len(q) > 10]  # Filter very short strings
        
        # Ensure we have 3-5 questions
        if len(questions) < 3:
            questions = [q.strip() for q in text.split('\n') if '?' in q and len(q.strip()) > 10][:5]
        
        return questions[:5]  # Max 5 questions

# ============================================================================
# 5. Prompt Generator Agent
# ============================================================================

class PromptGeneratorAgent:
    """Generates optimized prompts based on user idea and clarification answers"""
    
    def __init__(self, model_name: str = PROMPT_GENERATOR_MODEL):
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        self.system_prompt = """You are an expert prompt engineer. Based on the user's answers to the questions, create a comprehensive and well-structured prompt that addresses all the requirements. The prompt should be clear, specific, and optimized for achieving the user's goals. Include all relevant details about purpose, audience, tone, constraints, and desired output format. Return ONLY the created prompt text without any additional explanation. The prompt must be XML tagged. Below are the tags that you can use:
<role> (Mandatory Tag)
<context> (Mandatory Tag)
<instructions> (Mandatory Tag)
<skills> (Mandatory Tag)
<areas of knowledge> (Mandatory Tag)
<next steps> (Mandatory Tag)
<personality and style> (Mandatory Tag)
<output format> (Mandatory Tag)
<audience> (Optional)
<examples> (Optional)
<dont's> (Optional)"""
    
    def generate_prompt(self, user_idea: str, user_answers: Dict[str, str], 
                       feedback: str = "", stream_callback: Optional[Callable] = None) -> str:
        """Generate a prompt based on user idea and answers"""
        
        # Format user answers
        answers_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in user_answers.items()])
        
        if feedback:
            prompt = f"""{self.system_prompt}

Previous prompt feedback: {feedback}

User's original idea: {user_idea}

User's clarification answers:
{answers_text}

Based on the feedback, improve the prompt to address the issues mentioned. Return ONLY the improved prompt with XML tags."""
        else:
            prompt = f"""{self.system_prompt}

User's original idea: {user_idea}

User's clarification answers:
{answers_text}

Create a comprehensive prompt based on the above information. Return ONLY the prompt with XML tags."""
        
        try:
            full_text = ""
            if stream_callback:
                # Stream the response
                response = self.model.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        full_text += chunk.text
                        stream_callback(chunk.text, "prompt_generator")
                return full_text.strip()
            else:
                response = self.model.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            print(f"Error generating prompt: {e}")
            return ""

# ============================================================================
# 6. Prompt Tester Agent
# ============================================================================

class PromptTesterAgent:
    """Tests the generated prompt by running it on the model"""
    
    def __init__(self, model_name: str = PROMPT_TESTER_MODEL):
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
    
    def test_prompt(self, prompt: str, stream_callback: Optional[Callable] = None) -> str:
        """Run the prompt on the model and return the output"""
        try:
            # The prompt itself is the input to the model
            full_text = ""
            if stream_callback:
                # Stream the response
                response = self.model.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        full_text += chunk.text
                        stream_callback(chunk.text, "prompt_tester")
                return full_text.strip()
            else:
                response = self.model.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            print(f"Error testing prompt: {e}")
            return f"Error: {str(e)}"

# ============================================================================
# 7. Prompt Analyzer Agent
# ============================================================================

class PromptAnalyzerAgent:
    """Analyzes and scores prompts based on multiple parameters"""
    
    def __init__(self, model_name: str = ANALYZER_MODEL):
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        
        self.analysis_prompt_template = """You will take the prompt [{prompt}] and benchmark it extremely strictly against the following parameters:

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
Feedback: [Detailed feedback on how to improve the prompt]

Only output the score and feedback. No additional information is required."""
    
    def analyze(self, prompt: str, output: str, stream_callback: Optional[Callable] = None) -> Tuple[int, str]:
        """Analyze prompt and output, return score (0-100) and feedback"""
        analysis_prompt = self.analysis_prompt_template.format(
            prompt=prompt,
            output=output
        )
        
        try:
            full_text = ""
            if stream_callback:
                # Stream the response
                response = self.model.generate_content(analysis_prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        full_text += chunk.text
                        stream_callback(chunk.text, "analyzer")
                response_text = full_text.strip()
            else:
                response = self.model.generate_content(analysis_prompt)
                response_text = response.text.strip()
            
            # Parse score and feedback
            score, feedback = self._parse_response(response_text)
            return score, feedback
        except Exception as e:
            print(f"Error analyzing prompt: {e}")
            return 0, f"Error during analysis: {str(e)}"
    
    def _parse_response(self, response_text: str) -> Tuple[int, str]:
        """Parse score and feedback from analyzer response"""
        # Try to extract score
        score_match = re.search(r'Score:\s*(\d+)', response_text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
        else:
            # Try alternative patterns
            score_match = re.search(r'(\d+)\s*/\s*100', response_text)
            if score_match:
                score = int(score_match.group(1))
            else:
                score = 0
        
        # Extract feedback
        feedback_match = re.search(r'Feedback:\s*(.+?)(?=\n\n|$)', response_text, re.DOTALL | re.IGNORECASE)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        else:
            # Try to get everything after score
            parts = re.split(r'Score:\s*\d+', response_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                feedback = parts[1].strip()
            else:
                feedback = "No feedback provided."
        
        return score, feedback

# ============================================================================
# 8. Main Workflow Controller
# ============================================================================

class PromptOptimizerWorkflow:
    """Main workflow controller that orchestrates all agents"""
    
    def __init__(self):
        self.question_generator = QuestionGeneratorAgent()
        self.prompt_generator = PromptGeneratorAgent()
        self.prompt_tester = PromptTesterAgent()
        self.analyzer = PromptAnalyzerAgent()
        self.state = WorkflowState()
    
    def start_workflow(self, user_idea: str, stream_callback: Optional[Callable] = None) -> List[str]:
        """Start the workflow by generating clarification questions"""
        self.state.user_idea = user_idea
        self.state.clarification_questions = self.question_generator.generate_questions(user_idea, stream_callback)
        return self.state.clarification_questions
    
    def submit_answers(self, answers: Dict[str, str]):
        """Submit user answers to clarification questions"""
        self.state.user_answers = answers
    
    def run_optimization_loop(self, stream_callback: Optional[Callable] = None, 
                             status_callback: Optional[Callable] = None,
                             token_callback: Optional[Callable] = None) -> Dict:
        """Run the main optimization loop with streaming support"""
        results = {
            'final_prompt': '',
            'final_score': 0,
            'iterations': 0,
            'converged': False,
            'history': []
        }
        
        # Helper function to estimate tokens (rough estimate: ~4 chars per token)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        # Generate initial prompt
        feedback = ""
        
        for iteration in range(MAX_ITERATIONS):
            self.state.iteration = iteration + 1
            
            if status_callback:
                status_callback(f"Iteration {iteration + 1}/{MAX_ITERATIONS}: Generating prompt...")
            
            # Generate prompt
            if stream_callback:
                stream_callback(f"\n{'='*60}\n[Iteration {iteration + 1}/{MAX_ITERATIONS}]\n{'='*60}\n", "system")
                stream_callback("🤖 Prompt Generator: Generating optimized prompt...\n", "system")
            
            self.state.current_prompt = self.prompt_generator.generate_prompt(
                self.state.user_idea,
                self.state.user_answers,
                feedback,
                stream_callback
            )
            
            if token_callback:
                tokens = estimate_tokens(self.state.current_prompt)
                token_callback(time.time(), tokens, f"prompt_gen_iter_{iteration + 1}")
            
            if not self.state.current_prompt:
                if stream_callback:
                    stream_callback("❌ Error: Failed to generate prompt\n", "error")
                break
            
            if stream_callback:
                stream_callback(f"\n✅ Prompt generated ({len(self.state.current_prompt)} characters)\n", "system")
            
            # Test prompt
            if status_callback:
                status_callback(f"Iteration {iteration + 1}/{MAX_ITERATIONS}: Testing prompt...")
            
            if stream_callback:
                stream_callback("🧪 Prompt Tester: Running prompt on model...\n", "system")
            
            self.state.prompt_output = self.prompt_tester.test_prompt(
                self.state.current_prompt,
                stream_callback
            )
            
            if token_callback:
                tokens = estimate_tokens(self.state.prompt_output)
                token_callback(time.time(), tokens, f"prompt_test_iter_{iteration + 1}")
            
            if stream_callback:
                stream_callback(f"\n✅ Output generated ({len(self.state.prompt_output)} characters)\n", "system")
            
            # Analyze prompt and output
            if status_callback:
                status_callback(f"Iteration {iteration + 1}/{MAX_ITERATIONS}: Analyzing results...")
            
            if stream_callback:
                stream_callback("📊 Prompt Analyzer: Evaluating prompt and output...\n", "system")
            
            score, feedback = self.analyzer.analyze(
                self.state.current_prompt,
                self.state.prompt_output,
                stream_callback
            )
            
            if token_callback:
                tokens = estimate_tokens(feedback)
                token_callback(time.time(), tokens, f"analyzer_iter_{iteration + 1}")
            
            self.state.current_score = score
            self.state.feedback = feedback
            self.state.update_best()
            
            if stream_callback:
                stream_callback(f"\n📈 Score: {score}/100\n", "score")
                stream_callback(f"💬 Feedback: {feedback[:200]}...\n\n", "feedback")
            
            # Store iteration history
            results['history'].append({
                'iteration': iteration + 1,
                'score': score,
                'prompt': self.state.current_prompt,
                'output': self.state.prompt_output,
                'feedback': feedback
            })
            
            # Check if we've reached target score
            if score >= TARGET_SCORE:
                if stream_callback:
                    stream_callback(f"\n🎉 Target score achieved! ({score}/100)\n", "success")
                results['final_prompt'] = self.state.current_prompt
                results['final_score'] = score
                results['iterations'] = iteration + 1
                results['converged'] = True
                break
            
            # If this is the last iteration, use best prompt
            if iteration == MAX_ITERATIONS - 1:
                if stream_callback:
                    stream_callback(f"\n⚠️ Maximum iterations reached. Best score: {self.state.best_score}/100\n", "warning")
                results['final_prompt'] = self.state.best_prompt
                results['final_score'] = self.state.best_score
                results['iterations'] = MAX_ITERATIONS
                results['converged'] = False
        
        return results

# ============================================================================
# 9. Streamlit-Ready Functions
# ============================================================================

def initialize_workflow() -> PromptOptimizerWorkflow:
    """Initialize the workflow - call this at the start of Streamlit session"""
    return PromptOptimizerWorkflow()

def get_clarification_questions(workflow: PromptOptimizerWorkflow, user_idea: str, 
                                stream_callback: Optional[Callable] = None) -> List[str]:
    """Get clarification questions for user idea"""
    return workflow.start_workflow(user_idea, stream_callback)

def process_answers_and_optimize(workflow: PromptOptimizerWorkflow, answers: Dict[str, str],
                                stream_callback: Optional[Callable] = None,
                                status_callback: Optional[Callable] = None,
                                token_callback: Optional[Callable] = None) -> Dict:
    """Process user answers and run optimization loop"""
    workflow.submit_answers(answers)
    return workflow.run_optimization_loop(stream_callback, status_callback, token_callback)

# ============================================================================
# 10. Testing and Validation Functions
# ============================================================================

def test_question_generator():
    """Test the question generator"""
    agent = QuestionGeneratorAgent()
    questions = agent.generate_questions("I want to create a prompt for writing blog posts")
    print("Generated Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    return questions

def test_prompt_generator():
    """Test the prompt generator"""
    agent = PromptGeneratorAgent()
    user_idea = "I want to create a prompt for writing blog posts"
    answers = {
        "What is the main purpose?": "To write engaging blog posts about technology",
        "Who is the target audience?": "Tech enthusiasts and developers",
        "What tone should it have?": "Professional but friendly"
    }
    prompt = agent.generate_prompt(user_idea, answers)
    print("Generated Prompt:")
    print(prompt)
    return prompt

# ============================================================================
# 11. Example Usage (for command-line testing)
# ============================================================================

def run_example():
    """Example workflow execution"""
    
    # Initialize workflow
    workflow = PromptOptimizerWorkflow()
    
    # Step 1: User enters idea
    user_idea = input("Enter your prompt idea: ")
    
    # Step 2: Generate questions
    print("\nGenerating clarification questions...")
    questions = workflow.start_workflow(user_idea)
    
    print("\nPlease answer the following questions:\n")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    
    # Step 3: Collect answers
    print("\n" + "="*50)
    answers = {}
    for i, question in enumerate(questions, 1):
        answer = input(f"\nAnswer to question {i}: ")
        answers[question] = answer
    
    workflow.submit_answers(answers)
    
    # Step 4: Run optimization loop
    print("\n" + "="*50)
    print("Starting optimization loop...")
    print("="*50)
    
    results = workflow.run_optimization_loop()
    
    # Step 5: Display results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Final Score: {results['final_score']}/100")
    print(f"Iterations: {results['iterations']}")
    print(f"Converged: {results['converged']}")
    print("\nFinal Prompt:")
    print("-"*50)
    print(results['final_prompt'])
    
    return results

# ============================================================================
# Main Entry Point - Streamlit App
# ============================================================================

def main():
    """Main Streamlit application"""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. This is a Streamlit application.")
        return
    
    # Configure API key from Streamlit secrets
    if 'GOOGLE_API_KEY' in st.secrets:
        genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
    else:
        st.error("⚠️ Please configure GOOGLE_API_KEY in Streamlit secrets")
        st.stop()
    
    # Initialize session state
    if 'workflow' not in st.session_state:
        st.session_state.workflow = initialize_workflow()
        st.session_state.stage = 'idea'
        st.session_state.questions = []
        st.session_state.results = None
        st.session_state.terminal_output = ""
        st.session_state.token_data = []
        st.session_state.current_status = ""
        st.session_state.is_thinking = False
    
    # Stage 1: Get user idea
    if st.session_state.stage == 'idea':
        st.title("🧙 Prompt Optimizer")
        st.markdown("Welcome! I'm your prompt wizard. Let's create the perfect prompt together.")
        
        user_idea = st.text_area(
            'Enter your prompt idea:',
            placeholder="e.g., I want to create a prompt for writing engaging blog posts about technology...",
            height=100
        )
        
        if st.button('Generate Clarification Questions', type='primary'):
            if user_idea.strip():
                st.session_state.is_thinking = True
                st.session_state.token_data = []
                
                # Create layout for thinking status and ASCII animation
                status_placeholder = st.empty()
                animation_placeholder = st.empty()
                
                # ASCII animation frames
                spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                frame_index = 0
                
                def update_animation():
                    nonlocal frame_index
                    frame = spinner_frames[frame_index % len(spinner_frames)]
                    frame_index += 1
                    animation_text = f"""
```
{frame} Thinking...
    ╔═══════════════════════════════════════╗
    ║   Generating clarification questions  ║
    ║   Please wait while I analyze your    ║
    ║   prompt idea...                       ║
    ╚═══════════════════════════════════════╝
```
"""
                    animation_placeholder.markdown(animation_text)
                
                def token_callback(timestamp, tokens, stage):
                    st.session_state.token_data.append({
                        'timestamp': timestamp,
                        'tokens': tokens,
                        'stage': stage
                    })
                
                try:
                    with status_placeholder.status("Thinking... Generating clarification questions", state="running"):
                        # Show initial animation
                        update_animation()
                        
                        # Start animation in background thread
                        import threading
                        import time as time_module
                        stop_animation = threading.Event()
                        
                        def animate():
                            while not stop_animation.is_set():
                                try:
                                    update_animation()
                                    time_module.sleep(0.1)
                                except:
                                    break
                        
                        anim_thread = threading.Thread(target=animate, daemon=True)
                        anim_thread.start()
                        
                        try:
                            questions = get_clarification_questions(
                                st.session_state.workflow, 
                                user_idea,
                                None  # No stream callback
                            )
                        finally:
                            stop_animation.set()
                            time_module.sleep(0.2)  # Let animation stop
                        
                        st.session_state.questions = questions
                        st.session_state.is_thinking = False
                        st.session_state.stage = 'questions'
                        st.rerun()
                except Exception as e:
                    st.session_state.is_thinking = False
                    st.error(f"Error generating questions: {str(e)}")
            else:
                st.warning("Please enter a prompt idea first.")
    
    # Stage 2: Answer clarification questions
    elif st.session_state.stage == 'questions':
        st.title("📝 Answer Clarification Questions")
        st.markdown("Please answer the following questions to help me create the perfect prompt:")
        
        answers = {}
        for i, question in enumerate(st.session_state.questions, 1):
            answer = st.text_input(
                f'**Question {i}:** {question}',
                key=f'answer_{i}',
                placeholder="Your answer here..."
            )
            if answer:
                answers[question] = answer
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button('← Back to Idea'):
                st.session_state.stage = 'idea'
                st.rerun()
        
        with col2:
            if st.button('Submit Answers & Optimize →', type='primary'):
                if len(answers) == len(st.session_state.questions):
                    st.session_state.is_thinking = True
                    st.session_state.token_data = []
                    
                    # Create layout for thinking status and ASCII animation
                    status_placeholder = st.empty()
                    animation_placeholder = st.empty()
                    
                    # ASCII animation frames - more elaborate for optimization
                    spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                    animation_stages = [
                        "Analyzing your requirements...",
                        "Generating optimized prompt...",
                        "Testing prompt effectiveness...",
                        "Evaluating results...",
                        "Refining for perfection..."
                    ]
                    frame_index = 0
                    stage_index = 0
                    
                    def update_animation():
                        nonlocal frame_index, stage_index
                        frame = spinner_frames[frame_index % len(spinner_frames)]
                        frame_index += 1
                        # Change stage every 20 frames
                        if frame_index % 20 == 0:
                            stage_index = (stage_index + 1) % len(animation_stages)
                        
                        current_stage = animation_stages[stage_index]
                        animation_text = f"""
```
{frame} Thinking...
    ╔═══════════════════════════════════════╗
    ║   Optimizing Your Prompt              ║
    ║                                       ║
    ║   {current_stage:<37}║
    ║                                       ║
    ║   This may take a few minutes...      ║
    ╚═══════════════════════════════════════╝
    
    Progress: {'█' * ((frame_index % 40) // 4) + '░' * (10 - (frame_index % 40) // 4)}
```
"""
                        animation_placeholder.markdown(animation_text)
                    
                    def status_callback(status):
                        st.session_state.current_status = status
                        status_placeholder.status(f" {status}", state="running")
                    
                    def token_callback(timestamp, tokens, stage):
                        st.session_state.token_data.append({
                            'timestamp': timestamp,
                            'tokens': tokens,
                            'stage': stage
                        })
                    
                    try:
                        with status_placeholder.status("Thinking... Starting optimization", state="running"):
                            # Show initial animation
                            update_animation()
                            
                            # Start animation in background thread
                            import threading
                            import time as time_module
                            stop_animation = threading.Event()
                            
                            def animate():
                                while not stop_animation.is_set():
                                    try:
                                        update_animation()
                                        time_module.sleep(0.15)
                                    except:
                                        break
                            
                            anim_thread = threading.Thread(target=animate, daemon=True)
                            anim_thread.start()
                            
                            try:
                                results = process_answers_and_optimize(
                                    st.session_state.workflow, 
                                    answers,
                                    None,  # No stream callback
                                    status_callback,
                                    token_callback
                                )
                            finally:
                                stop_animation.set()
                                time_module.sleep(0.2)  # Let animation stop
                            
                            st.session_state.results = results
                            st.session_state.is_thinking = False
                            st.session_state.stage = 'results'
                            st.rerun()
                    except Exception as e:
                        st.session_state.is_thinking = False
                        st.error(f"Error during optimization: {str(e)}")
                else:
                    st.warning("Please answer all questions before submitting.")
    
    # Stage 3: Display results
    elif st.session_state.stage == 'results':
        st.title("✨ Final Optimized Prompt")
        
        results = st.session_state.results
        
        # Display score and metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Score", f"{results['final_score']}/100")
        with col2:
            st.metric("Iterations", results['iterations'])
        with col3:
            st.metric("Status", "✓ Converged" if results['converged'] else "⚠ Max Iterations")
        
        # Display final prompt
        st.subheader("📄 Your Optimized Prompt:")
        st.code(results['final_prompt'], language='xml')
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Prompt",
                data=results['final_prompt'],
                file_name="optimized_prompt.xml",
                mime="text/xml"
            )
        
        with col2:
            # Generate and download token usage graph
            if st.session_state.token_data and pd is not None and HAS_MATPLOTLIB:
                try:
                    df = pd.DataFrame(st.session_state.token_data)
                    df['cumulative'] = df['tokens'].cumsum()
                    
                    # Create time-based visualization
                    if len(df) > 1:
                        df['time_elapsed'] = (df['timestamp'] - df['timestamp'].iloc[0])
                        
                        # Create matplotlib figure
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df['time_elapsed'], df['cumulative'], linewidth=2, color='#1f77b4')
                        ax.fill_between(df['time_elapsed'], df['cumulative'], alpha=0.3, color='#1f77b4')
                        ax.set_xlabel('Time Elapsed (seconds)', fontsize=12)
                        ax.set_ylabel('Cumulative Tokens', fontsize=12)
                        ax.set_title('Token Usage Over Time', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        # Save to bytes buffer
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        plt.close(fig)
                        
                        st.download_button(
                            label="📊 Download Token Usage Graph",
                            data=buf,
                            file_name="token_usage_graph.png",
                            mime="image/png"
                        )
                    else:
                        st.info("📊 Token usage data available but insufficient for graph")
                except Exception as e:
                    st.warning(f"Could not generate token graph: {str(e)}")
            else:
                if not st.session_state.token_data:
                    st.info("📊 No token usage data available")
                elif not HAS_MATPLOTLIB:
                    st.info("📊 Install matplotlib to generate token usage graph")
        
        # Show iteration history in expander
        with st.expander("📊 View Iteration History"):
            for hist in results['history']:
                st.markdown(f"**Iteration {hist['iteration']}** - Score: {hist['score']}/100")
                with st.expander(f"View details for iteration {hist['iteration']}"):
                    st.markdown("**Prompt:**")
                    st.code(hist['prompt'], language='xml')
                    st.markdown("**Output:**")
                    st.text(hist['output'][:500] + "..." if len(hist['output']) > 500 else hist['output'])
                    st.markdown("**Feedback:**")
                    st.text(hist['feedback'])
        
        # Start over button
        if st.button('🔄 Start Over', type='primary'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()

