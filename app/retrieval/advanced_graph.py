"""Advanced graph-based retrieval with document relationships."""
from __future__ import annotations

import json
import time
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter

try:
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


@dataclass
class DocumentNode:
    """Represents a document node in the graph."""
    doc_id: str
    title: str
    content_summary: str
    topics: List[str]
    entities: List[str]
    creation_time: float
    last_accessed: float
    access_count: int = 0


@dataclass
class DocumentRelation:
    """Represents a relationship between documents."""
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    evidence: List[str]
    created_time: float


class AdvancedGraphRetriever:
    """Advanced graph-based retrieval with sophisticated relationships."""
    
    def __init__(self, storage_path: str = "./data/graph"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize graph
        if GRAPH_AVAILABLE:
            self.graph = nx.DiGraph()
            self.similarity_graph = nx.Graph()  # For similarity-based connections
        else:
            self.graph = None
            self.similarity_graph = None
            print("Warning: Graph functionality not available. Install networkx and sklearn.")
        
        # Storage files
        self.nodes_file = self.storage_path / "nodes.json"
        self.edges_file = self.storage_path / "edges.json"
        self.graph_file = self.storage_path / "graph.json"
        
        # Document storage
        self.nodes: Dict[str, DocumentNode] = {}
        self.relations: List[DocumentRelation] = []
        
        # Relation types and their properties
        self.relation_types = {
            'cites': {'weight': 0.9, 'directed': True, 'description': 'Document A cites document B'},
            'similar_content': {'weight': 0.7, 'directed': False, 'description': 'Documents have similar content'},
            'same_author': {'weight': 0.6, 'directed': False, 'description': 'Documents by same author'},
            'same_topic': {'weight': 0.8, 'directed': False, 'description': 'Documents about same topic'},
            'temporal_sequence': {'weight': 0.5, 'directed': True, 'description': 'Document A precedes document B temporally'},
            'supersedes': {'weight': 0.9, 'directed': True, 'description': 'Document A is superseded by document B'},
            'part_of': {'weight': 0.8, 'directed': True, 'description': 'Document A is part of document B'},
            'references': {'weight': 0.7, 'directed': True, 'description': 'Document A references document B'},
            'contradicts': {'weight': 0.6, 'directed': False, 'description': 'Documents have contradictory information'}
        }
        
        # Load existing data
        self.load_graph_data()
        
        # TF-IDF for content similarity
        if GRAPH_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.content_vectors = None
        
    def load_graph_data(self):
        """Load graph data from storage."""
        # Load nodes
        if self.nodes_file.exists():
            try:
                with open(self.nodes_file, 'r') as f:
                    nodes_data = json.load(f)
                    self.nodes = {
                        node_id: DocumentNode(**node_data) 
                        for node_id, node_data in nodes_data.items()
                    }
            except Exception as e:
                print(f"Failed to load nodes: {e}")
        
        # Load relations
        if self.edges_file.exists():
            try:
                with open(self.edges_file, 'r') as f:
                    relations_data = json.load(f)
                    self.relations = [DocumentRelation(**rel) for rel in relations_data]
            except Exception as e:
                print(f"Failed to load relations: {e}")
        
        # Rebuild graph from loaded data
        self._rebuild_graph()
    
    def save_graph_data(self):
        """Save graph data to storage."""
        try:
            # Save nodes
            nodes_data = {node_id: asdict(node) for node_id, node in self.nodes.items()}
            with open(self.nodes_file, 'w') as f:
                json.dump(nodes_data, f, indent=2)
            
            # Save relations
            relations_data = [asdict(rel) for rel in self.relations]
            with open(self.edges_file, 'w') as f:
                json.dump(relations_data, f, indent=2)
            
        except Exception as e:
            print(f"Failed to save graph data: {e}")
    
    def _rebuild_graph(self):
        """Rebuild NetworkX graph from stored data."""
        if not GRAPH_AVAILABLE:
            return
        
        # Clear existing graphs
        self.graph.clear()
        self.similarity_graph.clear()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id, **asdict(node))
            self.similarity_graph.add_node(node_id, **asdict(node))
        
        # Add edges
        for relation in self.relations:
            edge_data = {
                'relation_type': relation.relation_type,
                'strength': relation.strength,
                'evidence': relation.evidence,
                'created_time': relation.created_time
            }
            
            if self.relation_types[relation.relation_type]['directed']:
                self.graph.add_edge(relation.source_id, relation.target_id, **edge_data)
            else:
                self.graph.add_edge(relation.source_id, relation.target_id, **edge_data)
                self.similarity_graph.add_edge(relation.source_id, relation.target_id, **edge_data)
    
    def add_document(self, doc_id: str, title: str, content: str, 
                    metadata: Dict[str, Any] = None) -> DocumentNode:
        """Add a document to the graph."""
        # Extract topics and entities from content
        topics = self._extract_topics(content)
        entities = self._extract_entities(content)
        content_summary = content[:500] + "..." if len(content) > 500 else content
        
        node = DocumentNode(
            doc_id=doc_id,
            title=title,
            content_summary=content_summary,
            topics=topics,
            entities=entities,
            creation_time=time.time(),
            last_accessed=time.time(),
            access_count=0
        )
        
        self.nodes[doc_id] = node
        
        if GRAPH_AVAILABLE:
            self.graph.add_node(doc_id, **asdict(node))
            self.similarity_graph.add_node(doc_id, **asdict(node))
        
        # Automatically detect relationships with existing documents
        self._auto_detect_relationships(doc_id, content, metadata or {})
        
        # Save data
        self.save_graph_data()
        
        return node
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from document content."""
        # Simple keyword-based topic extraction
        # In production, would use more sophisticated NLP
        
        topic_keywords = {
            'machine_learning': ['machine learning', 'ml', 'neural network', 'deep learning', 'algorithm'],
            'data_science': ['data science', 'analytics', 'statistics', 'data analysis'],
            'technology': ['technology', 'software', 'hardware', 'computer', 'digital'],
            'business': ['business', 'company', 'market', 'strategy', 'revenue'],
            'research': ['research', 'study', 'experiment', 'analysis', 'findings'],
            'health': ['health', 'medical', 'healthcare', 'medicine', 'treatment'],
            'finance': ['finance', 'financial', 'investment', 'money', 'economic']
        }
        
        content_lower = content.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content."""
        # Simple pattern-based entity extraction
        # In production, would use NER models
        
        import re
        
        entities = []
        
        # Extract potential person names (Title Case sequences)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        entities.extend(re.findall(person_pattern, content))
        
        # Extract potential organization names (sequences with Corp, Inc, LLC, etc.)
        org_pattern = r'\b[A-Z][a-zA-Z\s]+ (?:Corp|Inc|LLC|Ltd|Company|Organization)\b'
        entities.extend(re.findall(org_pattern, content))
        
        # Extract dates
        date_pattern = r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b'
        entities.extend(re.findall(date_pattern, content))
        
        return list(set(entities))  # Remove duplicates
    
    def _auto_detect_relationships(self, new_doc_id: str, content: str, metadata: Dict[str, Any]):
        """Automatically detect relationships with existing documents."""
        if not GRAPH_AVAILABLE or new_doc_id not in self.nodes:
            return
        
        new_node = self.nodes[new_doc_id]
        
        for existing_id, existing_node in self.nodes.items():
            if existing_id == new_doc_id:
                continue
            
            # Detect various types of relationships
            relationships = []
            
            # 1. Topic similarity
            common_topics = set(new_node.topics) & set(existing_node.topics)
            if common_topics:
                strength = len(common_topics) / max(len(new_node.topics), len(existing_node.topics))
                relationships.append(('same_topic', strength, list(common_topics)))
            
            # 2. Entity overlap
            common_entities = set(new_node.entities) & set(existing_node.entities)
            if common_entities:
                strength = len(common_entities) / max(len(new_node.entities), len(existing_node.entities))
                relationships.append(('references', strength, list(common_entities)))
            
            # 3. Content similarity
            similarity_strength = self._calculate_content_similarity(
                new_node.content_summary, existing_node.content_summary
            )
            if similarity_strength > 0.3:
                relationships.append(('similar_content', similarity_strength, ['content_analysis']))
            
            # 4. Temporal relationship
            time_diff = abs(new_node.creation_time - existing_node.creation_time)
            if time_diff < 86400:  # Same day
                temporal_strength = max(0.3, 1.0 - time_diff / 86400)
                relationships.append(('temporal_sequence', temporal_strength, ['temporal_proximity']))
            
            # Add relationships that meet threshold
            for rel_type, strength, evidence in relationships:
                if strength > 0.4:  # Minimum threshold
                    self.add_relationship(new_doc_id, existing_id, rel_type, strength, evidence)
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity between two documents."""
        if not GRAPH_AVAILABLE:
            # Simple word overlap fallback
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0
        
        try:
            # Use TF-IDF similarity
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([content1, content2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            return 0.0
    
    def add_relationship(self, source_id: str, target_id: str, relation_type: str, 
                        strength: float, evidence: List[str]):
        """Add a relationship between documents."""
        if relation_type not in self.relation_types:
            print(f"Unknown relation type: {relation_type}")
            return
        
        # Check if relationship already exists
        existing = next((rel for rel in self.relations 
                        if rel.source_id == source_id and rel.target_id == target_id 
                        and rel.relation_type == relation_type), None)
        
        if existing:
            # Update existing relationship
            existing.strength = max(existing.strength, strength)
            existing.evidence.extend(evidence)
        else:
            # Create new relationship
            relation = DocumentRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=strength,
                evidence=evidence,
                created_time=time.time()
            )
            self.relations.append(relation)
            
            # Add to graphs
            if GRAPH_AVAILABLE:
                edge_data = {
                    'relation_type': relation_type,
                    'strength': strength,
                    'evidence': evidence,
                    'created_time': relation.created_time
                }
                
                if self.relation_types[relation_type]['directed']:
                    self.graph.add_edge(source_id, target_id, **edge_data)
                else:
                    self.graph.add_edge(source_id, target_id, **edge_data)
                    self.similarity_graph.add_edge(source_id, target_id, **edge_data)
    
    def search_by_graph_traversal(self, query: str, initial_docs: List[str], 
                                 max_hops: int = 2, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using graph traversal from initial document set."""
        if not GRAPH_AVAILABLE or not self.graph:
            return []
        
        # Start with initial documents
        candidates = set(initial_docs)
        scored_docs = {}
        
        # Score initial documents
        for doc_id in initial_docs:
            if doc_id in self.nodes:
                scored_docs[doc_id] = 1.0  # Full relevance for direct matches
                self.nodes[doc_id].last_accessed = time.time()
                self.nodes[doc_id].access_count += 1
        
        # Traverse graph for specified number of hops
        current_level = set(initial_docs)
        decay_factor = 0.7  # Relevance decay per hop
        
        for hop in range(max_hops):
            next_level = set()
            
            for doc_id in current_level:
                if doc_id not in self.graph:
                    continue
                
                # Get neighbors
                neighbors = list(self.graph.neighbors(doc_id))
                
                for neighbor_id in neighbors:
                    if neighbor_id in candidates:
                        continue  # Already processed
                    
                    # Calculate relevance score based on relationship strength
                    edge_data = self.graph[doc_id][neighbor_id]
                    relationship_strength = edge_data.get('strength', 0.5)
                    relation_weight = self.relation_types.get(
                        edge_data.get('relation_type', 'similar_content'), {}
                    ).get('weight', 0.5)
                    
                    # Score based on source relevance, relationship strength, and hop distance
                    source_relevance = scored_docs.get(doc_id, 0.0)
                    neighbor_score = source_relevance * relationship_strength * relation_weight * (decay_factor ** (hop + 1))
                    
                    # Query relevance boost
                    query_relevance = self._calculate_query_relevance(neighbor_id, query)
                    neighbor_score *= (1.0 + query_relevance)
                    
                    if neighbor_id in scored_docs:
                        scored_docs[neighbor_id] = max(scored_docs[neighbor_id], neighbor_score)
                    else:
                        scored_docs[neighbor_id] = neighbor_score
                    
                    candidates.add(neighbor_id)
                    next_level.add(neighbor_id)
            
            current_level = next_level
            if not current_level:
                break
        
        # Sort by score and return top results
        sorted_results = sorted(scored_docs.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def _calculate_query_relevance(self, doc_id: str, query: str) -> float:
        """Calculate how relevant a document is to the query."""
        if doc_id not in self.nodes:
            return 0.0
        
        node = self.nodes[doc_id]
        query_lower = query.lower()
        
        relevance = 0.0
        
        # Title match
        if any(word in node.title.lower() for word in query_lower.split()):
            relevance += 0.3
        
        # Topic match
        query_topics = self._extract_topics(query)
        common_topics = set(query_topics) & set(node.topics)
        if common_topics:
            relevance += 0.2 * len(common_topics) / len(query_topics) if query_topics else 0.2
        
        # Content match
        if any(word in node.content_summary.lower() for word in query_lower.split()):
            relevance += 0.2
        
        # Entity match
        if any(entity.lower() in query_lower for entity in node.entities):
            relevance += 0.3
        
        return min(1.0, relevance)
    
    def find_document_clusters(self, min_cluster_size: int = 2) -> List[List[str]]:
        """Find clusters of related documents."""
        if not GRAPH_AVAILABLE or not self.similarity_graph:
            return []
        
        try:
            # Use community detection
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(self.similarity_graph)
            
            # Filter by minimum size
            clusters = [list(community) for community in communities 
                       if len(community) >= min_cluster_size]
            
            return clusters
        except Exception:
            # Fallback: simple connected components
            return [list(component) for component in nx.connected_components(self.similarity_graph)
                   if len(component) >= min_cluster_size]
    
    def get_document_centrality(self, centrality_type: str = 'betweenness') -> Dict[str, float]:
        """Calculate centrality measures for documents."""
        if not GRAPH_AVAILABLE or not self.graph:
            return {}
        
        try:
            if centrality_type == 'betweenness':
                return nx.betweenness_centrality(self.graph)
            elif centrality_type == 'pagerank':
                return nx.pagerank(self.graph)
            elif centrality_type == 'eigenvector':
                return nx.eigenvector_centrality(self.graph)
            elif centrality_type == 'closeness':
                return nx.closeness_centrality(self.graph)
            else:
                return nx.degree_centrality(self.graph)
        except Exception:
            return {}
    
    def recommend_related_documents(self, doc_id: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Recommend documents related to a given document."""
        if not GRAPH_AVAILABLE or doc_id not in self.graph:
            return []
        
        recommendations = []
        
        # Direct neighbors with relationship details
        for neighbor in self.graph.neighbors(doc_id):
            edge_data = self.graph[doc_id][neighbor]
            strength = edge_data.get('strength', 0.5)
            relation_type = edge_data.get('relation_type', 'unknown')
            recommendations.append((neighbor, strength, relation_type))
        
        # Sort by strength and return top results
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def get_relationship_path(self, source_id: str, target_id: str) -> Optional[List[Tuple[str, str]]]:
        """Find the shortest path between two documents with relationship types."""
        if not GRAPH_AVAILABLE or not self.graph:
            return None
        
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            
            if len(path) < 2:
                return None
            
            # Get relationship types along the path
            path_with_relations = []
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                relation_type = edge_data.get('relation_type', 'unknown')
                path_with_relations.append((path[i + 1], relation_type))
            
            return path_with_relations
            
        except nx.NetworkXNoPath:
            return None
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the document graph."""
        stats = {
            'num_documents': len(self.nodes),
            'num_relationships': len(self.relations),
            'relationship_types': dict(Counter(rel.relation_type for rel in self.relations))
        }
        
        if GRAPH_AVAILABLE and self.graph:
            stats.update({
                'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph) if self.graph else 0,
                'num_connected_components': nx.number_connected_components(self.graph.to_undirected()),
                'graph_density': nx.density(self.graph),
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges()
            })
        
        return stats
    
    def export_graph_visualization(self, output_file: str = "graph_viz.json") -> Dict[str, Any]:
        """Export graph data for visualization."""
        viz_data = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes
        for doc_id, node in self.nodes.items():
            viz_data['nodes'].append({
                'id': doc_id,
                'title': node.title,
                'topics': node.topics,
                'access_count': node.access_count,
                'size': min(50, 10 + node.access_count * 2)  # Size based on popularity
            })
        
        # Add edges
        for relation in self.relations:
            viz_data['edges'].append({
                'source': relation.source_id,
                'target': relation.target_id,
                'type': relation.relation_type,
                'strength': relation.strength,
                'weight': relation.strength * 10  # For visualization
            })
        
        # Save to file if requested
        if output_file:
            viz_file = self.storage_path / output_file
            with open(viz_file, 'w') as f:
                json.dump(viz_data, f, indent=2)
        
        return viz_data
