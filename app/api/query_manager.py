"""Query cancellation and resource management system."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict


class QueryStatus(Enum):
    """Query execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ResourceType(Enum):
    """Resource types for tracking."""
    MEMORY = "memory"
    CPU = "cpu"
    EMBEDDING_CALLS = "embedding_calls"
    GENERATION_CALLS = "generation_calls"


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    max_memory_mb: float = 512.0
    max_cpu_percent: float = 50.0
    max_concurrent_queries: int = 10
    max_query_duration_seconds: int = 300
    max_embedding_calls_per_minute: int = 100
    max_generation_calls_per_minute: int = 50


@dataclass
class QueryMetadata:
    """Metadata for a query execution."""
    query_id: str
    user_id: Optional[str]
    query_text: str
    start_time: float
    end_time: Optional[float]
    status: QueryStatus
    resource_usage: Dict[str, float]
    cancellation_reason: Optional[str]
    priority: int = 1  # 1-10, higher is more important


@dataclass
class ResourceUsage:
    """Current resource usage tracking."""
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    active_queries: int = 0
    embedding_calls_per_minute: int = 0
    generation_calls_per_minute: int = 0


class QueryQueue:
    """Priority-based query queue with resource management."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self.waiting_queries: Dict[str, QueryMetadata] = {}
    
    async def enqueue(self, query_metadata: QueryMetadata) -> bool:
        """Add query to queue with priority."""
        try:
            # Higher priority number = higher priority (inverted for PriorityQueue)
            priority = -query_metadata.priority
            await self.queue.put((priority, query_metadata.start_time, query_metadata))
            self.waiting_queries[query_metadata.query_id] = query_metadata
            return True
        except asyncio.QueueFull:
            return False
    
    async def dequeue(self) -> Optional[QueryMetadata]:
        """Get next query from queue."""
        try:
            _, _, query_metadata = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            self.waiting_queries.pop(query_metadata.query_id, None)
            return query_metadata
        except asyncio.TimeoutError:
            return None
    
    def remove_query(self, query_id: str) -> bool:
        """Remove query from waiting queue (for cancellation)."""
        if query_id in self.waiting_queries:
            del self.waiting_queries[query_id]
            return True
        return False
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()


class ResourceMonitor:
    """Monitors and enforces resource usage limits."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.current_usage = ResourceUsage()
        self.call_timestamps: Dict[str, list] = defaultdict(list)
        self.lock = threading.Lock()
    
    def can_execute_query(self, estimated_resources: Dict[str, float] = None) -> tuple[bool, str]:
        """Check if query can be executed within resource limits."""
        with self.lock:
            estimated_resources = estimated_resources or {}
            
            # Check concurrent queries
            if self.current_usage.active_queries >= self.limits.max_concurrent_queries:
                return False, f"Max concurrent queries exceeded ({self.limits.max_concurrent_queries})"
            
            # Check memory
            estimated_memory = estimated_resources.get('memory_mb', 100.0)
            if self.current_usage.memory_mb + estimated_memory > self.limits.max_memory_mb:
                return False, f"Memory limit exceeded ({self.limits.max_memory_mb}MB)"
            
            # Check CPU
            estimated_cpu = estimated_resources.get('cpu_percent', 10.0)
            if self.current_usage.cpu_percent + estimated_cpu > self.limits.max_cpu_percent:
                return False, f"CPU limit exceeded ({self.limits.max_cpu_percent}%)"
            
            # Check rate limits
            current_time = time.time()
            minute_ago = current_time - 60
            
            # Clean old timestamps
            for call_type in list(self.call_timestamps.keys()):
                self.call_timestamps[call_type] = [
                    ts for ts in self.call_timestamps[call_type] if ts > minute_ago
                ]
            
            # Check embedding calls
            embedding_calls = len(self.call_timestamps.get('embedding', []))
            if embedding_calls >= self.limits.max_embedding_calls_per_minute:
                return False, f"Embedding API rate limit exceeded ({self.limits.max_embedding_calls_per_minute}/min)"
            
            # Check generation calls
            generation_calls = len(self.call_timestamps.get('generation', []))
            if generation_calls >= self.limits.max_generation_calls_per_minute:
                return False, f"Generation API rate limit exceeded ({self.limits.max_generation_calls_per_minute}/min)"
            
            return True, "OK"
    
    def allocate_resources(self, query_id: str, resources: Dict[str, float]):
        """Allocate resources for a query."""
        with self.lock:
            self.current_usage.memory_mb += resources.get('memory_mb', 0)
            self.current_usage.cpu_percent += resources.get('cpu_percent', 0)
            self.current_usage.active_queries += 1
    
    def release_resources(self, query_id: str, resources: Dict[str, float]):
        """Release resources after query completion."""
        with self.lock:
            self.current_usage.memory_mb = max(0, self.current_usage.memory_mb - resources.get('memory_mb', 0))
            self.current_usage.cpu_percent = max(0, self.current_usage.cpu_percent - resources.get('cpu_percent', 0))
            self.current_usage.active_queries = max(0, self.current_usage.active_queries - 1)
    
    def record_api_call(self, call_type: str):
        """Record an API call for rate limiting."""
        with self.lock:
            self.call_timestamps[call_type].append(time.time())
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        with self.lock:
            current_time = time.time()
            minute_ago = current_time - 60
            
            # Count API calls in last minute
            embedding_calls = len([ts for ts in self.call_timestamps.get('embedding', []) if ts > minute_ago])
            generation_calls = len([ts for ts in self.call_timestamps.get('generation', []) if ts > minute_ago])
            
            return ResourceUsage(
                memory_mb=self.current_usage.memory_mb,
                cpu_percent=self.current_usage.cpu_percent,
                active_queries=self.current_usage.active_queries,
                embedding_calls_per_minute=embedding_calls,
                generation_calls_per_minute=generation_calls
            )


class QueryManager:
    """Manages query execution with cancellation and resource management."""
    
    def __init__(self, resource_limits: ResourceLimits = None):
        self.resource_limits = resource_limits or ResourceLimits()
        self.resource_monitor = ResourceMonitor(self.resource_limits)
        self.query_queue = QueryQueue()
        
        # Active queries tracking
        self.active_queries: Dict[str, QueryMetadata] = {}
        self.query_tasks: Dict[str, asyncio.Task] = {}
        self.cancellation_events: Dict[str, asyncio.Event] = {}
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query_worker")
        
        # Start background worker
        self.worker_task = None
        self.shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the query manager background worker."""
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._background_worker())
    
    async def shutdown(self):
        """Shutdown the query manager."""
        self.shutdown_event.set()
        if self.worker_task:
            await self.worker_task
        self.thread_pool.shutdown(wait=True)
    
    async def submit_query(self, query_text: str, query_function: Callable, 
                          user_id: str = None, priority: int = 1,
                          estimated_resources: Dict[str, float] = None,
                          timeout_seconds: int = None) -> str:
        """Submit a query for execution with resource management."""
        
        query_id = str(uuid.uuid4())
        timeout_seconds = timeout_seconds or self.resource_limits.max_query_duration_seconds
        
        # Create query metadata
        query_metadata = QueryMetadata(
            query_id=query_id,
            user_id=user_id,
            query_text=query_text,
            start_time=time.time(),
            end_time=None,
            status=QueryStatus.PENDING,
            resource_usage={},
            cancellation_reason=None,
            priority=priority
        )
        
        # Check resource availability
        can_execute, reason = self.resource_monitor.can_execute_query(estimated_resources)
        if not can_execute:
            query_metadata.status = QueryStatus.FAILED
            query_metadata.cancellation_reason = f"Resource limit: {reason}"
            query_metadata.end_time = time.time()
            return query_id
        
        # Add to queue
        queued = await self.query_queue.enqueue(query_metadata)
        if not queued:
            query_metadata.status = QueryStatus.FAILED
            query_metadata.cancellation_reason = "Query queue full"
            query_metadata.end_time = time.time()
            return query_id
        
        # Store query function and create cancellation event
        self.query_tasks[query_id] = query_function
        self.cancellation_events[query_id] = asyncio.Event()
        
        return query_id
    
    async def cancel_query(self, query_id: str, reason: str = "User cancellation") -> bool:
        """Cancel a query execution."""
        
        # Check if query is in waiting queue
        if self.query_queue.remove_query(query_id):
            return True
        
        # Check if query is active
        if query_id in self.active_queries:
            query_metadata = self.active_queries[query_id]
            query_metadata.status = QueryStatus.CANCELLED
            query_metadata.cancellation_reason = reason
            query_metadata.end_time = time.time()
            
            # Signal cancellation
            if query_id in self.cancellation_events:
                self.cancellation_events[query_id].set()
            
            # Cancel asyncio task if exists
            if query_id in self.query_tasks and isinstance(self.query_tasks[query_id], asyncio.Task):
                self.query_tasks[query_id].cancel()
            
            return True
        
        return False
    
    async def get_query_status(self, query_id: str) -> Optional[QueryMetadata]:
        """Get current status of a query."""
        
        # Check waiting queue
        if query_id in self.query_queue.waiting_queries:
            return self.query_queue.waiting_queries[query_id]
        
        # Check active queries
        if query_id in self.active_queries:
            return self.active_queries[query_id]
        
        return None
    
    async def _background_worker(self):
        """Background worker that processes queued queries."""
        
        while not self.shutdown_event.is_set():
            try:
                # Get next query from queue
                query_metadata = await self.query_queue.dequeue()
                if query_metadata is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if query was cancelled while waiting
                if query_metadata.query_id in self.cancellation_events and self.cancellation_events[query_metadata.query_id].is_set():
                    continue
                
                # Execute query
                await self._execute_query(query_metadata)
                
            except Exception as e:
                print(f"Error in query worker: {e}")
                await asyncio.sleep(1)
    
    async def _execute_query(self, query_metadata: QueryMetadata):
        """Execute a single query with resource management and timeout."""
        
        query_id = query_metadata.query_id
        
        try:
            # Move to active queries
            query_metadata.status = QueryStatus.RUNNING
            self.active_queries[query_id] = query_metadata
            
            # Allocate resources
            estimated_resources = {'memory_mb': 100.0, 'cpu_percent': 10.0}
            self.resource_monitor.allocate_resources(query_id, estimated_resources)
            
            # Get query function
            query_function = self.query_tasks.get(query_id)
            if not query_function:
                raise ValueError(f"Query function not found for {query_id}")
            
            # Create timeout and cancellation handling
            cancellation_event = self.cancellation_events.get(query_id)
            timeout_seconds = self.resource_limits.max_query_duration_seconds
            
            # Execute query with timeout and cancellation
            try:
                result = await asyncio.wait_for(
                    self._run_query_with_cancellation(query_function, cancellation_event),
                    timeout=timeout_seconds
                )
                
                query_metadata.status = QueryStatus.COMPLETED
                query_metadata.end_time = time.time()
                
            except asyncio.TimeoutError:
                query_metadata.status = QueryStatus.TIMEOUT
                query_metadata.cancellation_reason = f"Query timeout after {timeout_seconds}s"
                query_metadata.end_time = time.time()
                
            except asyncio.CancelledError:
                query_metadata.status = QueryStatus.CANCELLED
                query_metadata.end_time = time.time()
            
        except Exception as e:
            query_metadata.status = QueryStatus.FAILED
            query_metadata.cancellation_reason = f"Execution error: {str(e)}"
            query_metadata.end_time = time.time()
        
        finally:
            # Release resources
            self.resource_monitor.release_resources(query_id, estimated_resources)
            
            # Cleanup
            self.active_queries.pop(query_id, None)
            self.query_tasks.pop(query_id, None)
            self.cancellation_events.pop(query_id, None)
    
    async def _run_query_with_cancellation(self, query_function: Callable, 
                                         cancellation_event: asyncio.Event):
        """Run query function with cancellation support."""
        
        if asyncio.iscoroutinefunction(query_function):
            # Async function - run with cancellation check
            while not cancellation_event.is_set():
                try:
                    return await asyncio.wait_for(query_function(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue  # Check cancellation and try again
        else:
            # Sync function - run in thread pool
            loop = asyncio.get_event_loop()
            while not cancellation_event.is_set():
                try:
                    return await asyncio.wait_for(
                        loop.run_in_executor(self.thread_pool, query_function),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue  # Check cancellation and try again
        
        raise asyncio.CancelledError("Query was cancelled")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        
        resource_usage = self.resource_monitor.get_current_usage()
        
        return {
            "resource_usage": asdict(resource_usage),
            "resource_limits": asdict(self.resource_limits),
            "queue_size": self.query_queue.get_queue_size(),
            "active_queries": len(self.active_queries),
            "total_capacity": self.resource_limits.max_concurrent_queries,
            "capacity_utilization": len(self.active_queries) / self.resource_limits.max_concurrent_queries,
            "active_query_ids": list(self.active_queries.keys())
        }
    
    def get_query_history(self, user_id: str = None, limit: int = 100) -> list[QueryMetadata]:
        """Get query history for a user or all users."""
        # In a production system, this would query a database
        # For now, return current active queries
        queries = list(self.active_queries.values())
        
        if user_id:
            queries = [q for q in queries if q.user_id == user_id]
        
        return sorted(queries, key=lambda q: q.start_time, reverse=True)[:limit]
