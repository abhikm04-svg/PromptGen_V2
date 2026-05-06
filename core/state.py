"""State management for PromptGen V2"""
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class WorkflowState:
    """Tracks the state of the workflow"""
    user_idea: str = ""
    clarification_questions: List[str] = field(default_factory=list)
    user_answers: Dict[str, str] = field(default_factory=dict)
    current_prompt: str = ""
    prompt_output: str = ""
    current_score: int = 0
    feedback: str = ""
    iteration: int = 0
    best_prompt: str = ""
    best_score: int = 0
    best_output: str = ""
    token_usage: List[Dict] = field(default_factory=list)

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
