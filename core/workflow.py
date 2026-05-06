"""Workflow controller for PromptGen V2"""
import time
from typing import Dict, List, Optional, Callable
from core.config import MAX_ITERATIONS, TARGET_SCORE
from core.state import WorkflowState
from core.agents import (
    QuestionGeneratorAgent, PromptGeneratorAgent,
    PromptTesterAgent, PromptAnalyzerAgent
)


class PromptOptimizerWorkflow:
    """Main workflow controller that orchestrates all agents"""

    def __init__(self):
        self.question_generator = QuestionGeneratorAgent()
        self.prompt_generator = PromptGeneratorAgent()
        self.prompt_tester = PromptTesterAgent()
        self.analyzer = PromptAnalyzerAgent()
        self.state = WorkflowState()

    def start_workflow(self, user_idea: str, stream_callback: Optional[Callable] = None) -> List[str]:
        self.state.user_idea = user_idea
        self.state.clarification_questions = self.question_generator.generate_questions(user_idea, stream_callback)
        return self.state.clarification_questions

    def submit_answers(self, answers: Dict[str, str]):
        self.state.user_answers = answers

    def run_optimization_loop(self, stream_callback: Optional[Callable] = None,
                              status_callback: Optional[Callable] = None,
                              token_callback: Optional[Callable] = None) -> Dict:
        results = {
            'final_prompt': '', 'final_score': 0,
            'iterations': 0, 'converged': False, 'history': []
        }

        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        feedback = ""

        for iteration in range(MAX_ITERATIONS):
            self.state.iteration = iteration + 1

            if status_callback:
                status_callback(f"Iteration {iteration + 1}/{MAX_ITERATIONS}: Generating prompt...")

            if stream_callback:
                stream_callback(f"\n{'='*60}\n[Iteration {iteration + 1}/{MAX_ITERATIONS}]\n{'='*60}\n", "system")
                stream_callback("🤖 Prompt Generator: Generating optimized prompt...\n", "system")

            self.state.current_prompt = self.prompt_generator.generate_prompt(
                self.state.user_idea, self.state.user_answers, feedback, stream_callback
            )

            if token_callback:
                token_callback(time.time(), estimate_tokens(self.state.current_prompt), f"prompt_gen_iter_{iteration + 1}")

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

            self.state.prompt_output = self.prompt_tester.test_prompt(self.state.current_prompt, stream_callback)

            if token_callback:
                token_callback(time.time(), estimate_tokens(self.state.prompt_output), f"prompt_test_iter_{iteration + 1}")

            if stream_callback:
                stream_callback(f"\n✅ Output generated ({len(self.state.prompt_output)} characters)\n", "system")

            # Analyze
            if status_callback:
                status_callback(f"Iteration {iteration + 1}/{MAX_ITERATIONS}: Analyzing results...")
            if stream_callback:
                stream_callback("📊 Prompt Analyzer: Evaluating prompt and output...\n", "system")

            score, feedback = self.analyzer.analyze(self.state.current_prompt, self.state.prompt_output, stream_callback)

            if token_callback:
                token_callback(time.time(), estimate_tokens(feedback), f"analyzer_iter_{iteration + 1}")

            self.state.current_score = score
            self.state.feedback = feedback
            self.state.update_best()

            if stream_callback:
                stream_callback(f"\n📈 Score: {score}/100\n", "score")
                preview = feedback[:200] + '...' if len(feedback) > 200 else feedback
                stream_callback(f"💬 Feedback: {preview}\n\n", "feedback")

            results['history'].append({
                'iteration': iteration + 1, 'score': score,
                'prompt': self.state.current_prompt,
                'output': self.state.prompt_output, 'feedback': feedback
            })

            if score >= TARGET_SCORE:
                if stream_callback:
                    stream_callback(f"\n🎉 Target score achieved! ({score}/100)\n", "success")
                results.update(final_prompt=self.state.current_prompt, final_score=score,
                               iterations=iteration + 1, converged=True)
                break

            if iteration == MAX_ITERATIONS - 1:
                if stream_callback:
                    stream_callback(f"\n⚠️ Maximum iterations reached. Best score: {self.state.best_score}/100\n", "warning")
                results.update(final_prompt=self.state.best_prompt, final_score=self.state.best_score,
                               iterations=MAX_ITERATIONS, converged=False)

        return results


def initialize_workflow() -> PromptOptimizerWorkflow:
    return PromptOptimizerWorkflow()

def get_clarification_questions(workflow, user_idea, stream_callback=None):
    return workflow.start_workflow(user_idea, stream_callback)

def process_answers_and_optimize(workflow, answers, stream_callback=None,
                                  status_callback=None, token_callback=None):
    workflow.submit_answers(answers)
    return workflow.run_optimization_loop(stream_callback, status_callback, token_callback)
