"""Multi-step reasoning and chain-of-thought generation."""
from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class ReasoningType(Enum):
    """Types of reasoning patterns."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    SEQUENTIAL = "sequential"


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process."""
    step_id: str
    step_type: ReasoningType
    premise: str
    conclusion: str
    confidence: float
    evidence: List[str]
    intermediate_result: Optional[str] = None
    depends_on: List[str] = None  # IDs of previous steps this depends on


@dataclass
class ReasoningChain:
    """Represents a complete chain of reasoning."""
    chain_id: str
    query: str
    steps: List[ReasoningStep]
    final_answer: str
    overall_confidence: float
    reasoning_type: ReasoningType
    execution_time_ms: int
    metadata: Dict[str, Any] = None


class MultiStepReasoner:
    """Handles multi-step reasoning and chain-of-thought generation."""
    
    def __init__(self):
        # Reasoning patterns and templates
        self.reasoning_patterns = self._load_reasoning_patterns()
        self.question_classifiers = self._load_question_classifiers()
        
        # Reasoning graph for complex dependencies
        self.reasoning_graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        
        # Confidence calibration
        self.confidence_weights = {
            'evidence_strength': 0.3,
            'logical_consistency': 0.3,
            'source_reliability': 0.2,
            'reasoning_complexity': 0.2
        }
    
    def _load_reasoning_patterns(self) -> Dict[ReasoningType, Dict[str, Any]]:
        """Load reasoning patterns and templates."""
        return {
            ReasoningType.DEDUCTIVE: {
                'pattern': 'If {premise1} and {premise2}, then {conclusion}',
                'keywords': ['all', 'every', 'therefore', 'thus', 'consequently'],
                'template': 'Given that {evidence}, we can deduce that {conclusion}'
            },
            ReasoningType.INDUCTIVE: {
                'pattern': 'Based on {examples}, we can generalize that {conclusion}',
                'keywords': ['most', 'generally', 'typically', 'usually', 'often'],
                'template': 'From the evidence {evidence}, we can infer that {conclusion}'
            },
            ReasoningType.ABDUCTIVE: {
                'pattern': 'The best explanation for {observation} is {hypothesis}',
                'keywords': ['best explanation', 'most likely', 'probably', 'suggests'],
                'template': 'Given {observation}, the most plausible explanation is {conclusion}'
            },
            ReasoningType.CAUSAL: {
                'pattern': '{cause} leads to {effect}',
                'keywords': ['causes', 'leads to', 'results in', 'due to', 'because'],
                'template': 'Because {cause}, we observe {effect}'
            },
            ReasoningType.COMPARATIVE: {
                'pattern': 'Compared to {baseline}, {subject} is {comparison}',
                'keywords': ['compared to', 'versus', 'unlike', 'similar to', 'different from'],
                'template': 'Comparing {subject1} and {subject2}, we find that {conclusion}'
            },
            ReasoningType.SEQUENTIAL: {
                'pattern': 'First {step1}, then {step2}, finally {step3}',
                'keywords': ['first', 'then', 'next', 'finally', 'step by step'],
                'template': 'The process involves: {steps}'
            }
        }
    
    def _load_question_classifiers(self) -> Dict[str, ReasoningType]:
        """Load question patterns to determine reasoning type."""
        return {
            r'why.*': ReasoningType.CAUSAL,
            r'how.*work.*': ReasoningType.SEQUENTIAL,
            r'what.*difference.*': ReasoningType.COMPARATIVE,
            r'compare.*': ReasoningType.COMPARATIVE,
            r'what.*cause.*': ReasoningType.CAUSAL,
            r'explain.*': ReasoningType.DEDUCTIVE,
            r'prove.*': ReasoningType.DEDUCTIVE,
            r'what.*evidence.*': ReasoningType.INDUCTIVE,
            r'what.*pattern.*': ReasoningType.INDUCTIVE,
            r'what.*likely.*': ReasoningType.ABDUCTIVE,
            r'what.*best.*explanation.*': ReasoningType.ABDUCTIVE,
        }
    
    async def reason_step_by_step(self, query: str, context_chunks: List[Dict[str, Any]], 
                                 generator) -> ReasoningChain:
        """Perform multi-step reasoning on a query."""
        start_time = time.time()
        
        # Step 1: Classify the reasoning type needed
        reasoning_type = self._classify_reasoning_type(query)
        
        # Step 2: Plan the reasoning steps
        reasoning_plan = self._plan_reasoning_steps(query, reasoning_type, context_chunks)
        
        # Step 3: Execute reasoning steps
        executed_steps = []
        for step_plan in reasoning_plan:
            executed_step = await self._execute_reasoning_step(
                step_plan, context_chunks, executed_steps, generator
            )
            executed_steps.append(executed_step)
        
        # Step 4: Synthesize final answer
        final_answer = await self._synthesize_final_answer(
            query, executed_steps, generator
        )
        
        # Step 5: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(executed_steps)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        chain = ReasoningChain(
            chain_id=f"reasoning_{int(time.time())}",
            query=query,
            steps=executed_steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence,
            reasoning_type=reasoning_type,
            execution_time_ms=execution_time,
            metadata={
                'num_steps': len(executed_steps),
                'context_chunks_used': len(context_chunks),
                'reasoning_complexity': self._assess_complexity(executed_steps)
            }
        )
        
        return chain
    
    def _classify_reasoning_type(self, query: str) -> ReasoningType:
        """Classify the type of reasoning needed for the query."""
        query_lower = query.lower()
        
        # Check against pattern classifiers
        for pattern, reasoning_type in self.question_classifiers.items():
            if re.search(pattern, query_lower):
                return reasoning_type
        
        # Check for keywords in reasoning patterns
        type_scores = {}
        for reasoning_type, pattern_info in self.reasoning_patterns.items():
            score = sum(1 for keyword in pattern_info['keywords'] 
                       if keyword in query_lower)
            type_scores[reasoning_type] = score
        
        # Return the type with highest score, default to deductive
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return ReasoningType.DEDUCTIVE  # Default
    
    def _plan_reasoning_steps(self, query: str, reasoning_type: ReasoningType, 
                            context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan the sequence of reasoning steps."""
        steps = []
        
        if reasoning_type == ReasoningType.DEDUCTIVE:
            steps = self._plan_deductive_steps(query, context_chunks)
        elif reasoning_type == ReasoningType.INDUCTIVE:
            steps = self._plan_inductive_steps(query, context_chunks)
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            steps = self._plan_abductive_steps(query, context_chunks)
        elif reasoning_type == ReasoningType.CAUSAL:
            steps = self._plan_causal_steps(query, context_chunks)
        elif reasoning_type == ReasoningType.COMPARATIVE:
            steps = self._plan_comparative_steps(query, context_chunks)
        elif reasoning_type == ReasoningType.SEQUENTIAL:
            steps = self._plan_sequential_steps(query, context_chunks)
        
        return steps
    
    def _plan_deductive_steps(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan deductive reasoning steps."""
        return [
            {
                'step_id': 'deductive_1',
                'step_type': ReasoningType.DEDUCTIVE,
                'purpose': 'Identify general principles or rules',
                'prompt_template': 'What general principles or rules can be extracted from this context: {context}?',
                'depends_on': []
            },
            {
                'step_id': 'deductive_2',
                'step_type': ReasoningType.DEDUCTIVE,
                'purpose': 'Apply principles to specific case',
                'prompt_template': 'Given the principles {previous_result} and the query "{query}", what specific conclusion can be drawn?',
                'depends_on': ['deductive_1']
            }
        ]
    
    def _plan_inductive_steps(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan inductive reasoning steps."""
        return [
            {
                'step_id': 'inductive_1',
                'step_type': ReasoningType.INDUCTIVE,
                'purpose': 'Identify specific examples or cases',
                'prompt_template': 'What specific examples, cases, or instances are mentioned in this context: {context}?',
                'depends_on': []
            },
            {
                'step_id': 'inductive_2',
                'step_type': ReasoningType.INDUCTIVE,
                'purpose': 'Find patterns across examples',
                'prompt_template': 'Looking at these examples {previous_result}, what patterns or common characteristics emerge?',
                'depends_on': ['inductive_1']
            },
            {
                'step_id': 'inductive_3',
                'step_type': ReasoningType.INDUCTIVE,
                'purpose': 'Generalize to answer query',
                'prompt_template': 'Based on the patterns {previous_result}, how can we answer "{query}"?',
                'depends_on': ['inductive_2']
            }
        ]
    
    def _plan_abductive_steps(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan abductive reasoning steps."""
        return [
            {
                'step_id': 'abductive_1',
                'step_type': ReasoningType.ABDUCTIVE,
                'purpose': 'Identify observations or phenomena',
                'prompt_template': 'What observations, phenomena, or facts are presented in this context: {context}?',
                'depends_on': []
            },
            {
                'step_id': 'abductive_2',
                'step_type': ReasoningType.ABDUCTIVE,
                'purpose': 'Generate possible explanations',
                'prompt_template': 'What are the possible explanations for these observations: {previous_result}?',
                'depends_on': ['abductive_1']
            },
            {
                'step_id': 'abductive_3',
                'step_type': ReasoningType.ABDUCTIVE,
                'purpose': 'Select best explanation',
                'prompt_template': 'Which explanation {previous_result} best answers "{query}" and why?',
                'depends_on': ['abductive_2']
            }
        ]
    
    def _plan_causal_steps(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan causal reasoning steps."""
        return [
            {
                'step_id': 'causal_1',
                'step_type': ReasoningType.CAUSAL,
                'purpose': 'Identify potential causes',
                'prompt_template': 'What potential causes, factors, or antecedents are mentioned in this context: {context}?',
                'depends_on': []
            },
            {
                'step_id': 'causal_2',
                'step_type': ReasoningType.CAUSAL,
                'purpose': 'Identify effects or outcomes',
                'prompt_template': 'What effects, outcomes, or consequences are described in relation to {previous_result}?',
                'depends_on': ['causal_1']
            },
            {
                'step_id': 'causal_3',
                'step_type': ReasoningType.CAUSAL,
                'purpose': 'Establish causal relationships',
                'prompt_template': 'What causal relationships can be established to answer "{query}" based on causes {previous_result}?',
                'depends_on': ['causal_2']
            }
        ]
    
    def _plan_comparative_steps(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan comparative reasoning steps."""
        return [
            {
                'step_id': 'comparative_1',
                'step_type': ReasoningType.COMPARATIVE,
                'purpose': 'Identify subjects to compare',
                'prompt_template': 'What are the subjects, items, or concepts that need to be compared based on "{query}" and this context: {context}?',
                'depends_on': []
            },
            {
                'step_id': 'comparative_2',
                'step_type': ReasoningType.COMPARATIVE,
                'purpose': 'Identify comparison dimensions',
                'prompt_template': 'What are the key dimensions, criteria, or attributes for comparing {previous_result}?',
                'depends_on': ['comparative_1']
            },
            {
                'step_id': 'comparative_3',
                'step_type': ReasoningType.COMPARATIVE,
                'purpose': 'Make comparisons and draw conclusions',
                'prompt_template': 'Compare the subjects along these dimensions {previous_result} and draw conclusions to answer "{query}".',
                'depends_on': ['comparative_2']
            }
        ]
    
    def _plan_sequential_steps(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan sequential reasoning steps."""
        return [
            {
                'step_id': 'sequential_1',
                'step_type': ReasoningType.SEQUENTIAL,
                'purpose': 'Identify process or sequence',
                'prompt_template': 'What process, sequence, or series of steps is described in this context: {context}?',
                'depends_on': []
            },
            {
                'step_id': 'sequential_2',
                'step_type': ReasoningType.SEQUENTIAL,
                'purpose': 'Order and detail steps',
                'prompt_template': 'Put these steps in proper order and provide details for each: {previous_result}',
                'depends_on': ['sequential_1']
            },
            {
                'step_id': 'sequential_3',
                'step_type': ReasoningType.SEQUENTIAL,
                'purpose': 'Apply sequence to query',
                'prompt_template': 'How does this sequence {previous_result} help answer "{query}"?',
                'depends_on': ['sequential_2']
            }
        ]
    
    async def _execute_reasoning_step(self, step_plan: Dict[str, Any], 
                                    context_chunks: List[Dict[str, Any]],
                                    previous_steps: List[ReasoningStep],
                                    generator) -> ReasoningStep:
        """Execute a single reasoning step."""
        
        # Prepare the prompt for this step
        prompt = self._prepare_step_prompt(step_plan, context_chunks, previous_steps)
        
        # Generate the reasoning for this step
        system_prompt = f"You are performing {step_plan['step_type'].value} reasoning. {step_plan['purpose']}. Be clear and logical."
        
        try:
            step_result = generator.complete(system=system_prompt, user=prompt)
        except Exception:
            step_result = f"Unable to complete reasoning step: {step_plan['purpose']}"
        
        # Extract evidence from context
        evidence = self._extract_evidence_for_step(step_plan, context_chunks)
        
        # Calculate confidence for this step
        step_confidence = self._calculate_step_confidence(step_result, evidence, step_plan)
        
        step = ReasoningStep(
            step_id=step_plan['step_id'],
            step_type=step_plan['step_type'],
            premise=prompt,
            conclusion=step_result,
            confidence=step_confidence,
            evidence=evidence,
            intermediate_result=step_result,
            depends_on=step_plan.get('depends_on', [])
        )
        
        return step
    
    def _prepare_step_prompt(self, step_plan: Dict[str, Any], 
                           context_chunks: List[Dict[str, Any]],
                           previous_steps: List[ReasoningStep]) -> str:
        """Prepare the prompt for a reasoning step."""
        
        # Get context text
        context_text = "\n\n".join([chunk.get('text', '') for chunk in context_chunks])
        
        # Get previous results if this step depends on others
        previous_result = ""
        if step_plan.get('depends_on'):
            dependent_steps = [step for step in previous_steps 
                             if step.step_id in step_plan['depends_on']]
            if dependent_steps:
                previous_result = "\n".join([step.intermediate_result for step in dependent_steps])
        
        # Format the prompt template
        prompt_template = step_plan['prompt_template']
        prompt = prompt_template.format(
            context=context_text[:2000],  # Limit context length
            previous_result=previous_result,
            query=step_plan.get('query', '')
        )
        
        return prompt
    
    def _extract_evidence_for_step(self, step_plan: Dict[str, Any], 
                                 context_chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract relevant evidence for a reasoning step."""
        evidence = []
        
        # Simple evidence extraction based on step type
        for chunk in context_chunks:
            chunk_text = chunk.get('text', '')
            if len(chunk_text) > 50:  # Only substantial chunks
                # Extract sentences that might serve as evidence
                sentences = re.split(r'[.!?]+', chunk_text)
                for sentence in sentences:
                    if len(sentence.strip()) > 20:
                        evidence.append(sentence.strip())
        
        # Limit evidence to avoid overwhelming
        return evidence[:5]
    
    def _calculate_step_confidence(self, step_result: str, evidence: List[str], 
                                 step_plan: Dict[str, Any]) -> float:
        """Calculate confidence for a reasoning step."""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Evidence strength
        if evidence:
            evidence_strength = min(0.3, len(evidence) * 0.1)
            confidence += evidence_strength
        
        # Factor 2: Result quality (length and coherence)
        if step_result and len(step_result) > 50:
            confidence += 0.1
        
        # Factor 3: Logical indicators in result
        logical_indicators = ['because', 'therefore', 'thus', 'since', 'given that', 'due to']
        logical_score = sum(1 for indicator in logical_indicators if indicator in step_result.lower())
        confidence += min(0.2, logical_score * 0.05)
        
        return max(0.1, min(0.95, confidence))
    
    async def _synthesize_final_answer(self, query: str, reasoning_steps: List[ReasoningStep], 
                                     generator) -> str:
        """Synthesize final answer from reasoning steps."""
        
        # Prepare synthesis prompt
        steps_summary = "\n".join([
            f"Step {i+1}: {step.conclusion}" 
            for i, step in enumerate(reasoning_steps)
        ])
        
        synthesis_prompt = f"""
Based on the following reasoning steps, provide a comprehensive answer to the query: "{query}"

Reasoning Steps:
{steps_summary}

Synthesize these steps into a coherent, well-reasoned answer that directly addresses the query.
"""
        
        system_prompt = "You are synthesizing a final answer from a chain of reasoning. Be comprehensive but concise."
        
        try:
            final_answer = generator.complete(system=system_prompt, user=synthesis_prompt)
        except Exception:
            # Fallback synthesis
            final_answer = f"Based on the reasoning steps, {reasoning_steps[-1].conclusion if reasoning_steps else 'no conclusion could be reached'}."
        
        return final_answer
    
    def _calculate_overall_confidence(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence for the reasoning chain."""
        if not reasoning_steps:
            return 0.1
        
        # Weighted average of step confidences
        step_confidences = [step.confidence for step in reasoning_steps]
        
        # Give more weight to later steps in the chain
        weights = [i + 1 for i in range(len(step_confidences))]
        weighted_sum = sum(conf * weight for conf, weight in zip(step_confidences, weights))
        total_weight = sum(weights)
        
        overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Apply complexity penalty for very long chains
        if len(reasoning_steps) > 5:
            complexity_penalty = (len(reasoning_steps) - 5) * 0.02
            overall_confidence = max(0.1, overall_confidence - complexity_penalty)
        
        return min(0.98, overall_confidence)
    
    def _assess_complexity(self, reasoning_steps: List[ReasoningStep]) -> str:
        """Assess the complexity of the reasoning chain."""
        num_steps = len(reasoning_steps)
        
        if num_steps <= 2:
            return "simple"
        elif num_steps <= 4:
            return "moderate"
        elif num_steps <= 6:
            return "complex"
        else:
            return "very_complex"
    
    async def stream_reasoning(self, query: str, context_chunks: List[Dict[str, Any]], 
                             generator) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream reasoning steps as they are computed."""
        
        # Classify and plan
        reasoning_type = self._classify_reasoning_type(query)
        reasoning_plan = self._plan_reasoning_steps(query, reasoning_type, context_chunks)
        
        yield {
            'event': 'reasoning_start',
            'reasoning_type': reasoning_type.value,
            'planned_steps': len(reasoning_plan),
            'timestamp': time.time()
        }
        
        executed_steps = []
        
        for i, step_plan in enumerate(reasoning_plan):
            yield {
                'event': 'step_start',
                'step_id': step_plan['step_id'],
                'step_number': i + 1,
                'purpose': step_plan['purpose'],
                'timestamp': time.time()
            }
            
            step = await self._execute_reasoning_step(
                step_plan, context_chunks, executed_steps, generator
            )
            executed_steps.append(step)
            
            yield {
                'event': 'step_complete',
                'step_id': step.step_id,
                'conclusion': step.conclusion,
                'confidence': step.confidence,
                'evidence_count': len(step.evidence),
                'timestamp': time.time()
            }
        
        # Synthesize final answer
        yield {
            'event': 'synthesis_start',
            'timestamp': time.time()
        }
        
        final_answer = await self._synthesize_final_answer(query, executed_steps, generator)
        overall_confidence = self._calculate_overall_confidence(executed_steps)
        
        yield {
            'event': 'reasoning_complete',
            'final_answer': final_answer,
            'overall_confidence': overall_confidence,
            'total_steps': len(executed_steps),
            'timestamp': time.time()
        }
    
    def get_reasoning_explanation(self, chain: ReasoningChain) -> str:
        """Generate a human-readable explanation of the reasoning process."""
        
        explanation = f"To answer '{chain.query}', I used {chain.reasoning_type.value} reasoning with {len(chain.steps)} steps:\n\n"
        
        for i, step in enumerate(chain.steps, 1):
            explanation += f"Step {i}: {step.conclusion}\n"
            if step.evidence:
                explanation += f"  Evidence: {'; '.join(step.evidence[:2])}\n"
            explanation += f"  Confidence: {step.confidence:.2f}\n\n"
        
        explanation += f"Final Answer: {chain.final_answer}\n"
        explanation += f"Overall Confidence: {chain.overall_confidence:.2f}"
        
        return explanation
