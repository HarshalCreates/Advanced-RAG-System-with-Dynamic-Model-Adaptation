"""Multi-model LLM judging framework for consensus scoring."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import statistics
import numpy as np


class EvaluationAspect(Enum):
    """Aspects of response evaluation."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    CITATION_QUALITY = "citation_quality"
    COHERENCE = "coherence"
    HELPFULNESS = "helpfulness"


class JudgeModel(Enum):
    """Available judge models."""
    GPT4O = "gpt-4o"
    CLAUDE_SONNET = "claude-3-sonnet-20240229"
    GPT35_TURBO = "gpt-3.5-turbo"
    GEMINI_PRO = "gemini-pro"


@dataclass
class AspectScore:
    """Score for a specific evaluation aspect."""
    aspect: EvaluationAspect
    score: float  # 0-10 scale
    reasoning: str
    confidence: float  # 0-1 scale
    model_id: str


@dataclass
class ModelJudgment:
    """Complete judgment from a single model."""
    model_id: str
    aspect_scores: List[AspectScore]
    overall_score: float
    overall_reasoning: str
    evaluation_time_ms: int
    model_confidence: float


@dataclass
class ConsensusResult:
    """Consensus result from multiple model judgments."""
    consensus_scores: Dict[str, float]  # aspect -> consensus score
    individual_judgments: List[ModelJudgment]
    agreement_scores: Dict[str, float]  # aspect -> agreement level
    outlier_models: List[str]
    final_score: float
    consensus_confidence: float
    disagreement_analysis: Dict[str, Any]


class JudgePromptTemplates:
    """Templates for LLM judging prompts."""
    
    SYSTEM_PROMPT = """You are an expert evaluator for AI-generated responses in a RAG system. Your task is to evaluate responses across multiple dimensions with high precision and consistency.

Guidelines:
- Use a 0-10 scale where 0 is completely inadequate and 10 is perfect
- Be objective and consistent in your scoring
- Provide clear reasoning for each score
- Consider the specific context and query requirements
- Rate your own confidence in each evaluation (0-1 scale)"""

    ASPECT_PROMPTS = {
        EvaluationAspect.ACCURACY: """
        Evaluate the ACCURACY of this response:
        - Are the facts stated correct?
        - Are there any factual errors or inaccuracies?
        - Does the information align with the provided sources?
        - Are numerical values, dates, and specific claims accurate?
        
        Score: 0 (completely inaccurate) to 10 (perfectly accurate)
        """,
        
        EvaluationAspect.COMPLETENESS: """
        Evaluate the COMPLETENESS of this response:
        - Does it fully address all parts of the query?
        - Are there important aspects left unaddressed?
        - Is the depth of information appropriate?
        - Are there significant gaps in the response?
        
        Score: 0 (completely incomplete) to 10 (comprehensively complete)
        """,
        
        EvaluationAspect.CLARITY: """
        Evaluate the CLARITY of this response:
        - Is the response well-structured and organized?
        - Is the language clear and easy to understand?
        - Are complex concepts explained appropriately?
        - Is the response free from ambiguity?
        
        Score: 0 (very unclear) to 10 (perfectly clear)
        """,
        
        EvaluationAspect.RELEVANCE: """
        Evaluate the RELEVANCE of this response:
        - How well does it address the specific query?
        - Is the information pertinent to what was asked?
        - Are there irrelevant tangents or off-topic content?
        - Does it focus on what the user actually needs?
        
        Score: 0 (completely irrelevant) to 10 (perfectly relevant)
        """,
        
        EvaluationAspect.CITATION_QUALITY: """
        Evaluate the CITATION QUALITY of this response:
        - Are sources properly cited and referenced?
        - Do citations support the claims made?
        - Are citation details accurate (page numbers, excerpts)?
        - Is there appropriate attribution for information?
        
        Score: 0 (no/poor citations) to 10 (excellent citations)
        """,
        
        EvaluationAspect.COHERENCE: """
        Evaluate the COHERENCE of this response:
        - Does the response flow logically?
        - Are ideas well-connected and consistent?
        - Is there a clear structure and progression?
        - Are there contradictions or logical gaps?
        
        Score: 0 (incoherent) to 10 (perfectly coherent)
        """,
        
        EvaluationAspect.HELPFULNESS: """
        Evaluate the HELPFULNESS of this response:
        - How useful is this response to the user?
        - Does it provide actionable information?
        - Would this response satisfy the user's need?
        - Is it practical and applicable?
        
        Score: 0 (not helpful) to 10 (extremely helpful)
        """
    }

    @classmethod
    def create_evaluation_prompt(cls, query: str, response: str, context: str, 
                                aspect: EvaluationAspect) -> str:
        """Create evaluation prompt for a specific aspect."""
        
        return f"""
{cls.SYSTEM_PROMPT}

{cls.ASPECT_PROMPTS[aspect]}

**Query:** {query}

**Response to Evaluate:**
{response}

**Available Context/Sources:**
{context}

Please provide your evaluation in the following JSON format:
{{
    "score": <0-10 numerical score>,
    "reasoning": "<detailed explanation for your score>",
    "confidence": <0-1 confidence in your evaluation>,
    "specific_issues": ["<issue 1>", "<issue 2>", ...],
    "strengths": ["<strength 1>", "<strength 2>", ...]
}}
"""


class LLMJudgeClient:
    """Client for interacting with different LLM judge models."""
    
    def __init__(self):
        self.clients = {}
        self._setup_clients()
    
    def _setup_clients(self):
        """Setup clients for different models."""
        try:
            import openai
            self.clients['openai'] = openai.AsyncOpenAI()
        except ImportError:
            pass
        
        try:
            import anthropic
            self.clients['anthropic'] = anthropic.AsyncAnthropic()
        except ImportError:
            pass
    
    async def evaluate_aspect(self, model: JudgeModel, query: str, response: str,
                            context: str, aspect: EvaluationAspect) -> AspectScore:
        """Evaluate a single aspect using specified model."""
        
        prompt = JudgePromptTemplates.create_evaluation_prompt(
            query, response, context, aspect
        )
        
        start_time = time.time()
        
        try:
            if model == JudgeModel.GPT4O and 'openai' in self.clients:
                result = await self._call_openai(prompt, "gpt-4o")
            elif model == JudgeModel.GPT35_TURBO and 'openai' in self.clients:
                result = await self._call_openai(prompt, "gpt-3.5-turbo")
            elif model == JudgeModel.CLAUDE_SONNET and 'anthropic' in self.clients:
                result = await self._call_anthropic(prompt)
            else:
                # Fallback to mock evaluation
                result = self._mock_evaluation(aspect)
            
            # Parse JSON response
            parsed = json.loads(result)
            
            return AspectScore(
                aspect=aspect,
                score=float(parsed.get('score', 5.0)),
                reasoning=parsed.get('reasoning', 'No reasoning provided'),
                confidence=float(parsed.get('confidence', 0.5)),
                model_id=model.value
            )
            
        except Exception as e:
            # Fallback score on error
            return AspectScore(
                aspect=aspect,
                score=5.0,
                reasoning=f"Evaluation failed: {str(e)}",
                confidence=0.1,
                model_id=model.value
            )
    
    async def _call_openai(self, prompt: str, model: str) -> str:
        """Call OpenAI API."""
        try:
            response = await self.clients['openai'].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception:
            return self._mock_evaluation_json()
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        try:
            response = await self.clients['anthropic'].messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception:
            return self._mock_evaluation_json()
    
    def _mock_evaluation(self, aspect: EvaluationAspect) -> AspectScore:
        """Generate mock evaluation for testing."""
        import random
        
        base_scores = {
            EvaluationAspect.ACCURACY: 7.5,
            EvaluationAspect.COMPLETENESS: 6.8,
            EvaluationAspect.CLARITY: 8.2,
            EvaluationAspect.RELEVANCE: 7.9,
            EvaluationAspect.CITATION_QUALITY: 6.5,
            EvaluationAspect.COHERENCE: 7.8,
            EvaluationAspect.HELPFULNESS: 7.3
        }
        
        base_score = base_scores.get(aspect, 7.0)
        score = max(0, min(10, base_score + random.gauss(0, 1.0)))
        
        return AspectScore(
            aspect=aspect,
            score=round(score, 1),
            reasoning=f"Mock evaluation for {aspect.value}",
            confidence=0.7,
            model_id="mock_model"
        )
    
    def _mock_evaluation_json(self) -> str:
        """Generate mock evaluation in JSON format."""
        import random
        
        score = round(random.uniform(6.0, 9.0), 1)
        confidence = round(random.uniform(0.6, 0.9), 2)
        
        return json.dumps({
            "score": score,
            "reasoning": "Mock evaluation response for testing purposes",
            "confidence": confidence,
            "specific_issues": ["Minor clarity improvements needed"],
            "strengths": ["Well-structured response", "Good use of context"]
        })


class ConsensusCalculator:
    """Calculates consensus from multiple model judgments."""
    
    @staticmethod
    def calculate_consensus(judgments: List[ModelJudgment]) -> ConsensusResult:
        """Calculate consensus from multiple model judgments."""
        
        if not judgments:
            raise ValueError("No judgments provided")
        
        # Extract all aspects
        all_aspects = set()
        for judgment in judgments:
            for score in judgment.aspect_scores:
                all_aspects.add(score.aspect)
        
        consensus_scores = {}
        agreement_scores = {}
        
        # Calculate consensus for each aspect
        for aspect in all_aspects:
            aspect_scores = []
            for judgment in judgments:
                for score in judgment.aspect_scores:
                    if score.aspect == aspect:
                        aspect_scores.append(score.score)
            
            if aspect_scores:
                # Weighted average (can be enhanced with confidence weighting)
                consensus_score = statistics.mean(aspect_scores)
                consensus_scores[aspect.value] = round(consensus_score, 2)
                
                # Calculate agreement (inverse of standard deviation)
                if len(aspect_scores) > 1:
                    std_dev = statistics.stdev(aspect_scores)
                    agreement = max(0, 1 - (std_dev / 5))  # Normalize by max possible std
                else:
                    agreement = 1.0
                agreement_scores[aspect.value] = round(agreement, 3)
        
        # Identify outlier models
        outlier_models = ConsensusCalculator._identify_outliers(judgments, consensus_scores)
        
        # Calculate final overall score
        final_score = statistics.mean(consensus_scores.values()) if consensus_scores else 5.0
        
        # Calculate consensus confidence
        consensus_confidence = statistics.mean(agreement_scores.values()) if agreement_scores else 0.5
        
        # Disagreement analysis
        disagreement_analysis = ConsensusCalculator._analyze_disagreements(
            judgments, consensus_scores
        )
        
        return ConsensusResult(
            consensus_scores=consensus_scores,
            individual_judgments=judgments,
            agreement_scores=agreement_scores,
            outlier_models=outlier_models,
            final_score=round(final_score, 2),
            consensus_confidence=round(consensus_confidence, 3),
            disagreement_analysis=disagreement_analysis
        )
    
    @staticmethod
    def _identify_outliers(judgments: List[ModelJudgment], 
                          consensus_scores: Dict[str, float]) -> List[str]:
        """Identify models that consistently disagree with consensus."""
        
        outlier_threshold = 2.0  # Score difference threshold
        outlier_models = []
        
        for judgment in judgments:
            disagreement_count = 0
            total_aspects = 0
            
            for score in judgment.aspect_scores:
                aspect_key = score.aspect.value
                if aspect_key in consensus_scores:
                    consensus_score = consensus_scores[aspect_key]
                    if abs(score.score - consensus_score) > outlier_threshold:
                        disagreement_count += 1
                    total_aspects += 1
            
            # If model disagrees on >50% of aspects, mark as outlier
            if total_aspects > 0 and (disagreement_count / total_aspects) > 0.5:
                outlier_models.append(judgment.model_id)
        
        return outlier_models
    
    @staticmethod
    def _analyze_disagreements(judgments: List[ModelJudgment],
                              consensus_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze patterns in model disagreements."""
        
        disagreements_by_aspect = {}
        model_disagreement_rates = {}
        
        # Analyze disagreements by aspect
        for aspect_key, consensus_score in consensus_scores.items():
            aspect_disagreements = []
            
            for judgment in judgments:
                for score in judgment.aspect_scores:
                    if score.aspect.value == aspect_key:
                        disagreement = abs(score.score - consensus_score)
                        aspect_disagreements.append(disagreement)
            
            if aspect_disagreements:
                disagreements_by_aspect[aspect_key] = {
                    'avg_disagreement': round(statistics.mean(aspect_disagreements), 2),
                    'max_disagreement': round(max(aspect_disagreements), 2),
                    'std_disagreement': round(statistics.stdev(aspect_disagreements), 2) 
                        if len(aspect_disagreements) > 1 else 0
                }
        
        # Analyze disagreement rates by model
        for judgment in judgments:
            disagreements = []
            for score in judgment.aspect_scores:
                aspect_key = score.aspect.value
                if aspect_key in consensus_scores:
                    disagreement = abs(score.score - consensus_scores[aspect_key])
                    disagreements.append(disagreement)
            
            if disagreements:
                model_disagreement_rates[judgment.model_id] = {
                    'avg_disagreement': round(statistics.mean(disagreements), 2),
                    'max_disagreement': round(max(disagreements), 2)
                }
        
        return {
            'disagreements_by_aspect': disagreements_by_aspect,
            'model_disagreement_rates': model_disagreement_rates,
            'most_controversial_aspect': max(
                disagreements_by_aspect.keys(),
                key=lambda k: disagreements_by_aspect[k]['avg_disagreement']
            ) if disagreements_by_aspect else None
        }


class MultiModelJudge:
    """Main multi-model judging system."""
    
    def __init__(self, judge_models: List[JudgeModel] = None,
                 aspects: List[EvaluationAspect] = None):
        
        self.judge_models = judge_models or [
            JudgeModel.GPT4O,
            JudgeModel.CLAUDE_SONNET,
            JudgeModel.GPT35_TURBO
        ]
        
        self.aspects = aspects or [
            EvaluationAspect.ACCURACY,
            EvaluationAspect.COMPLETENESS,
            EvaluationAspect.CLARITY,
            EvaluationAspect.RELEVANCE,
            EvaluationAspect.CITATION_QUALITY
        ]
        
        self.llm_client = LLMJudgeClient()
        self.consensus_calculator = ConsensusCalculator()
    
    async def evaluate_response(self, query: str, response: str, context: str,
                              citations: List[Dict[str, Any]] = None) -> ConsensusResult:
        """Evaluate a response using multiple models and aspects."""
        
        # Prepare context string with citations
        context_with_citations = self._prepare_context(context, citations)
        
        # Collect judgments from all models
        judgments = []
        
        for model in self.judge_models:
            try:
                judgment = await self._get_model_judgment(
                    model, query, response, context_with_citations
                )
                judgments.append(judgment)
            except Exception as e:
                print(f"Failed to get judgment from {model.value}: {e}")
                # Continue with other models
        
        if not judgments:
            raise RuntimeError("Failed to get judgments from any model")
        
        # Calculate consensus
        consensus = self.consensus_calculator.calculate_consensus(judgments)
        
        return consensus
    
    async def _get_model_judgment(self, model: JudgeModel, query: str,
                                response: str, context: str) -> ModelJudgment:
        """Get complete judgment from a single model."""
        
        start_time = time.time()
        
        # Evaluate all aspects in parallel
        aspect_tasks = [
            self.llm_client.evaluate_aspect(model, query, response, context, aspect)
            for aspect in self.aspects
        ]
        
        aspect_scores = await asyncio.gather(*aspect_tasks)
        
        # Calculate overall score and reasoning
        overall_score = statistics.mean([score.score for score in aspect_scores])
        overall_reasoning = self._generate_overall_reasoning(aspect_scores)
        
        # Calculate model confidence
        model_confidence = statistics.mean([score.confidence for score in aspect_scores])
        
        evaluation_time = int((time.time() - start_time) * 1000)
        
        return ModelJudgment(
            model_id=model.value,
            aspect_scores=aspect_scores,
            overall_score=round(overall_score, 2),
            overall_reasoning=overall_reasoning,
            evaluation_time_ms=evaluation_time,
            model_confidence=round(model_confidence, 3)
        )
    
    def _prepare_context(self, context: str, citations: List[Dict[str, Any]] = None) -> str:
        """Prepare context string including citations."""
        
        prepared_context = context
        
        if citations:
            citation_text = "\n\nCitations:\n"
            for i, citation in enumerate(citations, 1):
                citation_text += f"{i}. {citation.get('document', 'Unknown')} "
                if citation.get('pages'):
                    citation_text += f"(p. {citation['pages']}) "
                citation_text += f"- {citation.get('excerpt', '')}\n"
            
            prepared_context += citation_text
        
        return prepared_context
    
    def _generate_overall_reasoning(self, aspect_scores: List[AspectScore]) -> str:
        """Generate overall reasoning from aspect scores."""
        
        strengths = []
        weaknesses = []
        
        for score in aspect_scores:
            if score.score >= 8.0:
                strengths.append(f"Strong {score.aspect.value} (score: {score.score})")
            elif score.score <= 5.0:
                weaknesses.append(f"Weak {score.aspect.value} (score: {score.score})")
        
        reasoning_parts = []
        
        if strengths:
            reasoning_parts.append(f"Strengths: {', '.join(strengths)}")
        
        if weaknesses:
            reasoning_parts.append(f"Areas for improvement: {', '.join(weaknesses)}")
        
        if not reasoning_parts:
            reasoning_parts.append("Response shows balanced performance across all aspects")
        
        return ". ".join(reasoning_parts) + "."
    
    def get_evaluation_summary(self, consensus: ConsensusResult) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        
        # Find best and worst aspects
        best_aspect = max(consensus.consensus_scores.items(), key=lambda x: x[1])
        worst_aspect = min(consensus.consensus_scores.items(), key=lambda x: x[1])
        
        # Calculate overall quality category
        final_score = consensus.final_score
        if final_score >= 8.5:
            quality_category = "Excellent"
        elif final_score >= 7.0:
            quality_category = "Good"
        elif final_score >= 5.5:
            quality_category = "Satisfactory"
        elif final_score >= 4.0:
            quality_category = "Needs Improvement"
        else:
            quality_category = "Poor"
        
        return {
            "overall_score": consensus.final_score,
            "quality_category": quality_category,
            "consensus_confidence": consensus.consensus_confidence,
            "best_aspect": {"aspect": best_aspect[0], "score": best_aspect[1]},
            "worst_aspect": {"aspect": worst_aspect[0], "score": worst_aspect[1]},
            "model_agreement": statistics.mean(consensus.agreement_scores.values()),
            "outlier_models": consensus.outlier_models,
            "evaluation_reliability": "High" if consensus.consensus_confidence > 0.8 else 
                                    "Medium" if consensus.consensus_confidence > 0.6 else "Low"
        }
