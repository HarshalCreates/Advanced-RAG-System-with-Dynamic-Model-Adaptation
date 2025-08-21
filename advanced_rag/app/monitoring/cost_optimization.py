"""Real-time cost tracking and optimization system."""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path
import uuid


class CostCategory(Enum):
    """Categories of costs."""
    COMPUTE = "compute"
    API_CALLS = "api_calls"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    EXTERNAL_SERVICES = "external_services"


class OptimizationStrategy(Enum):
    """Cost optimization strategies."""
    CACHE_RESULTS = "cache_results"
    BATCH_REQUESTS = "batch_requests"
    REDUCE_MODEL_CALLS = "reduce_model_calls"
    OPTIMIZE_RETRIEVAL = "optimize_retrieval"
    COMPRESS_DATA = "compress_data"
    SCALE_DOWN = "scale_down"
    USE_CHEAPER_MODEL = "use_cheaper_model"


@dataclass
class CostEntry:
    """Individual cost tracking entry."""
    entry_id: str
    timestamp: float
    user_id: Optional[str]
    operation_type: str  # "query", "ingestion", "storage", etc.
    cost_category: CostCategory
    cost_usd: float
    resource_usage: Dict[str, float]  # CPU seconds, memory MB-hours, API calls, etc.
    metadata: Dict[str, Any]


@dataclass
class CostBudget:
    """Cost budget configuration."""
    budget_id: str
    name: str
    period: str  # "daily", "weekly", "monthly"
    limit_usd: float
    current_spend_usd: float
    alert_thresholds: List[float]  # Alert at 50%, 80%, 95%
    start_date: float
    end_date: float
    is_active: bool


@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation."""
    recommendation_id: str
    strategy: OptimizationStrategy
    description: str
    estimated_savings_usd: float
    estimated_savings_percent: float
    implementation_effort: str  # "low", "medium", "high"
    priority: int  # 1-10
    applicable_operations: List[str]
    metadata: Dict[str, Any]


class CostCalculator:
    """Calculates costs for different operations and resources."""
    
    def __init__(self):
        # Pricing models (simplified - in production, fetch from actual providers)
        self.pricing = {
            "openai": {
                "gpt-4o": {"input_tokens": 0.00005, "output_tokens": 0.00015},  # per token
                "text-embedding-ada-002": {"tokens": 0.0000001}  # per token
            },
            "anthropic": {
                "claude-3-sonnet": {"input_tokens": 0.000003, "output_tokens": 0.000015}
            },
            "cohere": {
                "embed-english-v2.0": {"tokens": 0.0000001}
            },
            "compute": {
                "cpu_hour": 0.048,  # per CPU hour
                "memory_gb_hour": 0.0067,  # per GB hour
                "storage_gb_month": 0.023  # per GB month
            },
            "bandwidth": {
                "gb_transfer": 0.09  # per GB
            }
        }
    
    def calculate_query_cost(self, query_data: Dict[str, Any]) -> float:
        """Calculate cost for a query operation."""
        total_cost = 0.0
        
        # Embedding cost
        embedding_backend = query_data.get("embedding_backend", "hash")
        embedding_model = query_data.get("embedding_model", "fallback")
        query_tokens = len(query_data.get("query", "").split()) * 1.3  # Rough token estimate
        
        if embedding_backend in self.pricing:
            model_pricing = self.pricing[embedding_backend].get(embedding_model, {})
            token_cost = model_pricing.get("tokens", 0)
            total_cost += query_tokens * token_cost
        
        # Generation cost
        generation_backend = query_data.get("generation_backend", "echo")
        generation_model = query_data.get("generation_model", "fallback")
        
        if generation_backend in self.pricing:
            model_pricing = self.pricing[generation_backend].get(generation_model, {})
            input_tokens = query_tokens + query_data.get("context_tokens", 500)  # Query + context
            output_tokens = query_data.get("response_tokens", 100)  # Estimated response length
            
            total_cost += input_tokens * model_pricing.get("input_tokens", 0)
            total_cost += output_tokens * model_pricing.get("output_tokens", 0)
        
        # Compute cost
        response_time_seconds = query_data.get("response_time_ms", 1000) / 1000
        cpu_cores = query_data.get("cpu_cores", 0.1)  # Fraction of CPU core used
        memory_gb = query_data.get("memory_gb", 0.1)  # Memory used in GB
        
        compute_cost = (
            (response_time_seconds / 3600) * cpu_cores * self.pricing["compute"]["cpu_hour"] +
            (response_time_seconds / 3600) * memory_gb * self.pricing["compute"]["memory_gb_hour"]
        )
        total_cost += compute_cost
        
        return round(total_cost, 6)
    
    def calculate_ingestion_cost(self, ingestion_data: Dict[str, Any]) -> float:
        """Calculate cost for document ingestion."""
        total_cost = 0.0
        
        # Embedding cost for document chunks
        chunk_count = ingestion_data.get("chunk_count", 0)
        avg_tokens_per_chunk = ingestion_data.get("avg_tokens_per_chunk", 200)
        embedding_backend = ingestion_data.get("embedding_backend", "hash")
        embedding_model = ingestion_data.get("embedding_model", "fallback")
        
        if embedding_backend in self.pricing:
            model_pricing = self.pricing[embedding_backend].get(embedding_model, {})
            token_cost = model_pricing.get("tokens", 0)
            total_cost += chunk_count * avg_tokens_per_chunk * token_cost
        
        # Storage cost (per month, prorated)
        document_size_gb = ingestion_data.get("document_size_gb", 0.001)
        storage_cost_per_day = document_size_gb * self.pricing["compute"]["storage_gb_month"] / 30
        total_cost += storage_cost_per_day
        
        # Processing compute cost
        processing_time_seconds = ingestion_data.get("processing_time_seconds", 10)
        cpu_cores = 0.5  # Ingestion is more CPU intensive
        memory_gb = 0.2
        
        compute_cost = (
            (processing_time_seconds / 3600) * cpu_cores * self.pricing["compute"]["cpu_hour"] +
            (processing_time_seconds / 3600) * memory_gb * self.pricing["compute"]["memory_gb_hour"]
        )
        total_cost += compute_cost
        
        return round(total_cost, 6)
    
    def calculate_storage_cost(self, storage_data: Dict[str, Any]) -> float:
        """Calculate ongoing storage costs."""
        total_gb = storage_data.get("total_storage_gb", 0)
        days_stored = storage_data.get("days_stored", 1)
        
        daily_cost = total_gb * self.pricing["compute"]["storage_gb_month"] / 30
        return round(daily_cost * days_stored, 6)


class CostOptimizer:
    """Analyzes costs and provides optimization recommendations."""
    
    def __init__(self):
        self.optimization_rules = self._create_optimization_rules()
    
    def _create_optimization_rules(self) -> List[Dict[str, Any]]:
        """Create cost optimization rules."""
        return [
            {
                "condition": lambda costs: self._get_category_cost(costs, CostCategory.API_CALLS) > 0.10,
                "strategy": OptimizationStrategy.CACHE_RESULTS,
                "description": "Implement result caching to reduce API calls",
                "savings_percent": 30,
                "effort": "medium"
            },
            {
                "condition": lambda costs: self._get_repeated_queries_ratio(costs) > 0.3,
                "strategy": OptimizationStrategy.CACHE_RESULTS,
                "description": "Cache frequently repeated queries",
                "savings_percent": 25,
                "effort": "low"
            },
            {
                "condition": lambda costs: self._get_category_cost(costs, CostCategory.COMPUTE) > 0.05,
                "strategy": OptimizationStrategy.OPTIMIZE_RETRIEVAL,
                "description": "Optimize retrieval algorithms to reduce compute time",
                "savings_percent": 20,
                "effort": "high"
            },
            {
                "condition": lambda costs: self._get_small_batch_ratio(costs) > 0.5,
                "strategy": OptimizationStrategy.BATCH_REQUESTS,
                "description": "Batch small requests to improve efficiency",
                "savings_percent": 15,
                "effort": "medium"
            },
            {
                "condition": lambda costs: self._get_expensive_model_usage(costs) > 0.8,
                "strategy": OptimizationStrategy.USE_CHEAPER_MODEL,
                "description": "Use less expensive models for simple queries",
                "savings_percent": 40,
                "effort": "low"
            }
        ]
    
    def _get_category_cost(self, costs: List[CostEntry], category: CostCategory) -> float:
        """Get total cost for a category."""
        return sum(cost.cost_usd for cost in costs if cost.cost_category == category)
    
    def _get_repeated_queries_ratio(self, costs: List[CostEntry]) -> float:
        """Get ratio of repeated queries."""
        query_costs = [cost for cost in costs if cost.operation_type == "query"]
        if not query_costs:
            return 0.0
        
        query_texts = {}
        for cost in query_costs:
            query_text = cost.metadata.get("query_text", "")
            query_texts[query_text] = query_texts.get(query_text, 0) + 1
        
        repeated_queries = sum(1 for count in query_texts.values() if count > 1)
        return repeated_queries / len(query_texts) if query_texts else 0.0
    
    def _get_small_batch_ratio(self, costs: List[CostEntry]) -> float:
        """Get ratio of small batch operations."""
        total_operations = len(costs)
        if total_operations == 0:
            return 0.0
        
        small_operations = sum(1 for cost in costs if cost.cost_usd < 0.001)
        return small_operations / total_operations
    
    def _get_expensive_model_usage(self, costs: List[CostEntry]) -> float:
        """Get ratio of expensive model usage."""
        api_costs = [cost for cost in costs if cost.cost_category == CostCategory.API_CALLS]
        if not api_costs:
            return 0.0
        
        expensive_threshold = 0.01  # $0.01 per operation
        expensive_operations = sum(1 for cost in api_costs if cost.cost_usd > expensive_threshold)
        return expensive_operations / len(api_costs)
    
    def analyze_costs(self, costs: List[CostEntry]) -> List[CostOptimizationRecommendation]:
        """Analyze costs and generate optimization recommendations."""
        recommendations = []
        
        total_cost = sum(cost.cost_usd for cost in costs)
        
        for rule in self.optimization_rules:
            if rule["condition"](costs):
                estimated_savings = total_cost * (rule["savings_percent"] / 100)
                
                recommendation = CostOptimizationRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    strategy=rule["strategy"],
                    description=rule["description"],
                    estimated_savings_usd=estimated_savings,
                    estimated_savings_percent=rule["savings_percent"],
                    implementation_effort=rule["effort"],
                    priority=self._calculate_priority(estimated_savings, rule["effort"]),
                    applicable_operations=self._get_applicable_operations(costs, rule["strategy"]),
                    metadata={}
                )
                recommendations.append(recommendation)
        
        # Sort by priority (highest first)
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations
    
    def _calculate_priority(self, savings_usd: float, effort: str) -> int:
        """Calculate priority based on savings and implementation effort."""
        effort_multipliers = {"low": 1.0, "medium": 0.7, "high": 0.4}
        effort_multiplier = effort_multipliers.get(effort, 0.5)
        
        # Priority = savings impact * ease of implementation
        priority_score = savings_usd * 1000 * effort_multiplier  # Scale up for integer priority
        return min(max(int(priority_score), 1), 10)
    
    def _get_applicable_operations(self, costs: List[CostEntry], strategy: OptimizationStrategy) -> List[str]:
        """Get operations that would benefit from a strategy."""
        if strategy == OptimizationStrategy.CACHE_RESULTS:
            return ["query", "embedding_generation"]
        elif strategy == OptimizationStrategy.BATCH_REQUESTS:
            return ["ingestion", "embedding_generation"]
        elif strategy == OptimizationStrategy.USE_CHEAPER_MODEL:
            return ["query", "generation"]
        else:
            return list(set(cost.operation_type for cost in costs))


class CostTrackingManager:
    """Main cost tracking and optimization system."""
    
    def __init__(self, storage_path: str = "./data/cost_tracking"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.costs_file = self.storage_path / "cost_entries.json"
        self.budgets_file = self.storage_path / "cost_budgets.json"
        
        # Components
        self.cost_calculator = CostCalculator()
        self.cost_optimizer = CostOptimizer()
        
        # Data storage
        self.cost_entries: List[CostEntry] = []
        self.budgets: Dict[str, CostBudget] = {}
        
        self.load_data()
        self.setup_default_budgets()
    
    def load_data(self):
        """Load cost tracking data."""
        # Load cost entries
        if self.costs_file.exists():
            try:
                with open(self.costs_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data[-10000:]:  # Keep last 10k entries
                        entry_data['cost_category'] = CostCategory(entry_data['cost_category'])
                        entry = CostEntry(**entry_data)
                        self.cost_entries.append(entry)
            except Exception as e:
                print(f"Failed to load cost entries: {e}")
        
        # Load budgets
        if self.budgets_file.exists():
            try:
                with open(self.budgets_file, 'r') as f:
                    data = json.load(f)
                    for budget_data in data:
                        budget = CostBudget(**budget_data)
                        self.budgets[budget.budget_id] = budget
            except Exception as e:
                print(f"Failed to load budgets: {e}")
    
    def save_data(self):
        """Save cost tracking data."""
        try:
            # Save cost entries (keep last 10k)
            entries_data = []
            for entry in self.cost_entries[-10000:]:
                data = asdict(entry)
                data['cost_category'] = entry.cost_category.value
                entries_data.append(data)
            
            with open(self.costs_file, 'w') as f:
                json.dump(entries_data, f, indent=2)
            
            # Save budgets
            budgets_data = [asdict(budget) for budget in self.budgets.values()]
            with open(self.budgets_file, 'w') as f:
                json.dump(budgets_data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save cost tracking data: {e}")
    
    def setup_default_budgets(self):
        """Setup default cost budgets if none exist."""
        if not self.budgets:
            current_time = time.time()
            
            # Daily budget
            daily_budget = CostBudget(
                budget_id="daily_default",
                name="Daily Cost Budget",
                period="daily",
                limit_usd=10.0,
                current_spend_usd=0.0,
                alert_thresholds=[0.5, 0.8, 0.95],
                start_date=current_time,
                end_date=current_time + 86400,  # 24 hours
                is_active=True
            )
            
            # Monthly budget
            monthly_budget = CostBudget(
                budget_id="monthly_default",
                name="Monthly Cost Budget",
                period="monthly",
                limit_usd=200.0,
                current_spend_usd=0.0,
                alert_thresholds=[0.5, 0.8, 0.95],
                start_date=current_time,
                end_date=current_time + (30 * 86400),  # 30 days
                is_active=True
            )
            
            self.budgets[daily_budget.budget_id] = daily_budget
            self.budgets[monthly_budget.budget_id] = monthly_budget
            
            self.save_data()
    
    def record_query_cost(self, query_data: Dict[str, Any], user_id: str = None) -> str:
        """Record cost for a query operation."""
        cost_usd = self.cost_calculator.calculate_query_cost(query_data)
        
        entry = CostEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=time.time(),
            user_id=user_id,
            operation_type="query",
            cost_category=CostCategory.API_CALLS,
            cost_usd=cost_usd,
            resource_usage={
                "response_time_ms": query_data.get("response_time_ms", 0),
                "cpu_cores": query_data.get("cpu_cores", 0.1),
                "memory_gb": query_data.get("memory_gb", 0.1)
            },
            metadata=query_data
        )
        
        self.cost_entries.append(entry)
        self._update_budgets(cost_usd)
        
        # Save periodically
        if len(self.cost_entries) % 50 == 0:
            self.save_data()
        
        return entry.entry_id
    
    def record_ingestion_cost(self, ingestion_data: Dict[str, Any], user_id: str = None) -> str:
        """Record cost for document ingestion."""
        cost_usd = self.cost_calculator.calculate_ingestion_cost(ingestion_data)
        
        entry = CostEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=time.time(),
            user_id=user_id,
            operation_type="ingestion",
            cost_category=CostCategory.API_CALLS,
            cost_usd=cost_usd,
            resource_usage={
                "chunk_count": ingestion_data.get("chunk_count", 0),
                "processing_time_seconds": ingestion_data.get("processing_time_seconds", 0),
                "document_size_gb": ingestion_data.get("document_size_gb", 0)
            },
            metadata=ingestion_data
        )
        
        self.cost_entries.append(entry)
        self._update_budgets(cost_usd)
        
        return entry.entry_id
    
    def _update_budgets(self, cost_usd: float):
        """Update budget spend and check alerts."""
        current_time = time.time()
        
        for budget in self.budgets.values():
            if not budget.is_active:
                continue
            
            # Check if budget period is still valid
            if current_time > budget.end_date:
                self._reset_budget(budget)
            
            # Update spend
            budget.current_spend_usd += cost_usd
            
            # Check alerts
            spend_ratio = budget.current_spend_usd / budget.limit_usd
            for threshold in budget.alert_thresholds:
                if spend_ratio >= threshold:
                    self._trigger_budget_alert(budget, spend_ratio)
    
    def _reset_budget(self, budget: CostBudget):
        """Reset budget for new period."""
        current_time = time.time()
        
        if budget.period == "daily":
            budget.start_date = current_time
            budget.end_date = current_time + 86400
        elif budget.period == "weekly":
            budget.start_date = current_time
            budget.end_date = current_time + (7 * 86400)
        elif budget.period == "monthly":
            budget.start_date = current_time
            budget.end_date = current_time + (30 * 86400)
        
        budget.current_spend_usd = 0.0
    
    def _trigger_budget_alert(self, budget: CostBudget, spend_ratio: float):
        """Trigger budget alert (integrate with alerting system)."""
        # In production, this would integrate with the alerting system
        print(f"🚨 Budget Alert: {budget.name} at {spend_ratio:.1%} of limit (${budget.current_spend_usd:.2f}/${budget.limit_usd:.2f})")
    
    def get_cost_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost summary for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_costs = [entry for entry in self.cost_entries if entry.timestamp > cutoff_time]
        
        if not recent_costs:
            return {"total_cost_usd": 0.0, "entries": 0}
        
        # Calculate totals by category
        category_costs = {}
        for category in CostCategory:
            category_costs[category.value] = sum(
                entry.cost_usd for entry in recent_costs if entry.cost_category == category
            )
        
        # Calculate totals by operation
        operation_costs = {}
        for entry in recent_costs:
            operation_costs[entry.operation_type] = operation_costs.get(entry.operation_type, 0) + entry.cost_usd
        
        # Calculate user costs
        user_costs = {}
        for entry in recent_costs:
            if entry.user_id:
                user_costs[entry.user_id] = user_costs.get(entry.user_id, 0) + entry.cost_usd
        
        total_cost = sum(entry.cost_usd for entry in recent_costs)
        
        return {
            "time_period_hours": hours,
            "total_cost_usd": round(total_cost, 4),
            "total_entries": len(recent_costs),
            "cost_by_category": {k: round(v, 4) for k, v in category_costs.items()},
            "cost_by_operation": {k: round(v, 4) for k, v in operation_costs.items()},
            "cost_by_user": {k: round(v, 4) for k, v in sorted(user_costs.items(), key=lambda x: x[1], reverse=True)[:10]},
            "average_cost_per_operation": round(total_cost / len(recent_costs), 6) if recent_costs else 0,
            "projected_daily_cost": round(total_cost * (24 / hours), 4) if hours != 24 else round(total_cost, 4),
            "projected_monthly_cost": round(total_cost * (24 * 30 / hours), 2)
        }
    
    def get_optimization_recommendations(self, days: int = 7) -> List[CostOptimizationRecommendation]:
        """Get cost optimization recommendations."""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_costs = [entry for entry in self.cost_entries if entry.timestamp > cutoff_time]
        
        return self.cost_optimizer.analyze_costs(recent_costs)
    
    def get_budget_status(self) -> List[Dict[str, Any]]:
        """Get status of all budgets."""
        budget_statuses = []
        
        for budget in self.budgets.values():
            spend_ratio = budget.current_spend_usd / budget.limit_usd if budget.limit_usd > 0 else 0
            
            status = "green"
            if spend_ratio >= 0.95:
                status = "red"
            elif spend_ratio >= 0.8:
                status = "yellow"
            elif spend_ratio >= 0.5:
                status = "orange"
            
            budget_statuses.append({
                "budget_id": budget.budget_id,
                "name": budget.name,
                "period": budget.period,
                "current_spend_usd": round(budget.current_spend_usd, 4),
                "limit_usd": budget.limit_usd,
                "spend_ratio": round(spend_ratio, 3),
                "remaining_usd": round(budget.limit_usd - budget.current_spend_usd, 4),
                "status": status,
                "days_remaining": max(0, int((budget.end_date - time.time()) / 86400)),
                "is_active": budget.is_active
            })
        
        return budget_statuses
    
    def estimate_query_cost(self, query_data: Dict[str, Any]) -> float:
        """Estimate cost for a query before execution."""
        return self.cost_calculator.calculate_query_cost(query_data)
    
    def get_cost_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get cost trends over time."""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_costs = [entry for entry in self.cost_entries if entry.timestamp > cutoff_time]
        
        if not recent_costs:
            return {"daily_costs": [], "trend": "stable"}
        
        # Group by day
        daily_costs = {}
        for entry in recent_costs:
            day = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
            daily_costs[day] = daily_costs.get(day, 0) + entry.cost_usd
        
        # Calculate trend
        daily_values = list(daily_costs.values())
        if len(daily_values) >= 2:
            recent_avg = sum(daily_values[-7:]) / len(daily_values[-7:])  # Last week
            earlier_avg = sum(daily_values[:-7]) / len(daily_values[:-7]) if len(daily_values) > 7 else recent_avg
            
            if recent_avg > earlier_avg * 1.1:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "daily_costs": [{"date": k, "cost_usd": round(v, 4)} for k, v in sorted(daily_costs.items())],
            "trend": trend,
            "total_cost_usd": round(sum(daily_values), 4),
            "average_daily_cost": round(sum(daily_values) / len(daily_values), 4),
            "peak_daily_cost": round(max(daily_values), 4) if daily_values else 0,
            "lowest_daily_cost": round(min(daily_values), 4) if daily_values else 0
        }
