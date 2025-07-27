import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod
import random
from collections import defaultdict, deque

class WalkStrategy(Enum):
    RANDOM = "random"
    BIASED = "biased"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"

@dataclass
class WalkConfig:
    strategy: WalkStrategy = WalkStrategy.RANDOM
    max_steps: int = 10
    temperature: float = 1.0
    return_param: float = 1.0  # node2vec p parameter
    inout_param: float = 1.0   # node2vec q parameter
    semantic_threshold: float = 0.7
    use_memory: bool = True
    memory_window: int = 5

@dataclass
class WalkStep:
    node: str
    edge_type: Optional[str]
    timestamp: float
    context: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class GraphWalker(ABC):
    """Abstract base class for different graph walking strategies"""
    
    @abstractmethod
    def next_step(self, current_node: str, graph: nx.Graph, 
                  history: List[WalkStep], config: WalkConfig) -> Optional[str]:
        pass

class RandomWalker(GraphWalker):
    def next_step(self, current_node: str, graph: nx.Graph, 
                  history: List[WalkStep], config: WalkConfig) -> Optional[str]:
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            return None
        return random.choice(neighbors)

class BiasedWalker(GraphWalker):
    def next_step(self, current_node: str, graph: nx.Graph, 
                  history: List[WalkStep], config: WalkConfig) -> Optional[str]:
        if len(history) < 2:
            return RandomWalker().next_step(current_node, graph, history, config)
        
        prev_node = history[-2].node
        neighbors = list(graph.neighbors(current_node))
        
        if not neighbors:
            return None
        
        # Calculate transition probabilities based on node2vec
        probs = []
        for neighbor in neighbors:
            if neighbor == prev_node:
                # Return to previous node
                prob = 1.0 / config.return_param
            elif graph.has_edge(prev_node, neighbor):
                # Neighbor is also connected to previous node
                prob = 1.0
            else:
                # Exploring further away
                prob = 1.0 / config.inout_param
            probs.append(prob)
        
        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]
        
        return np.random.choice(neighbors, p=probs)

class SemanticWalker(GraphWalker):
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
    
    def next_step(self, current_node: str, graph: nx.Graph, 
                  history: List[WalkStep], config: WalkConfig) -> Optional[str]:
        neighbors = list(graph.neighbors(current_node))
        if not neighbors or not self.embedding_model:
            return RandomWalker().next_step(current_node, graph, history, config)
        
        # Get semantic context from recent history
        context_nodes = [step.node for step in history[-config.memory_window:]]
        
        # Calculate semantic similarity scores
        similarities = []
        for neighbor in neighbors:
            sim_score = self._calculate_semantic_similarity(
                neighbor, context_nodes, graph
            )
            similarities.append(sim_score)
        
        # Apply temperature scaling
        if config.temperature > 0:
            exp_sims = np.exp(np.array(similarities) / config.temperature)
            probs = exp_sims / np.sum(exp_sims)
            return np.random.choice(neighbors, p=probs)
        else:
            return neighbors[np.argmax(similarities)]
    
    def _calculate_semantic_similarity(self, node: str, context_nodes: List[str], 
                                     graph: nx.Graph) -> float:
        # Placeholder for semantic similarity calculation
        # In practice, this would use embeddings
        return random.random()

class EnhancedWalkLLM:
    def __init__(self, llm_client, config: WalkConfig):
        self.llm_client = llm_client
        self.config = config
        self.walkers = {
            WalkStrategy.RANDOM: RandomWalker(),
            WalkStrategy.BIASED: BiasedWalker(),
            WalkStrategy.SEMANTIC: SemanticWalker(),
        }
        self.walk_history = []
        self.prompt_templates = self._load_prompt_templates()
        
    def _load_prompt_templates(self) -> Dict[str, str]:
        return {
            "exploration": """
Based on the knowledge graph walk path: {walk_path}
Current position: {current_node}
Context: {context}

Generate insights or ask questions that explore the connections between these concepts.
Consider the semantic relationships and potential implications.
""",
            "reasoning": """
Following the reasoning path through the knowledge graph:
{walk_path}

Based on this traversal, what logical conclusions can you draw?
What patterns or relationships emerge from this path?
""",
            "creative": """
Inspired by this journey through connected concepts:
{walk_path}

Create something new - a story, hypothesis, or creative connection that bridges these ideas.
Let your imagination follow the semantic trail.
"""
        }
    
    def perform_walk(self, graph: nx.Graph, start_node: str, 
                    prompt_type: str = "exploration") -> Dict[str, Any]:
        """Perform a complete walk with LLM integration"""
        
        walk_path = []
        current_node = start_node
        
        # Initialize walk
        initial_step = WalkStep(
            node=current_node,
            edge_type=None,
            timestamp=0.0,
            context=self._get_node_context(current_node, graph)
        )
        walk_path.append(initial_step)
        
        # Perform walk steps
        walker = self.walkers[self.config.strategy]
        
        for step in range(self.config.max_steps):
            next_node = walker.next_step(current_node, graph, walk_path, self.config)
            
            if next_node is None:
                break
                
            # Get edge information
            edge_data = graph.get_edge_data(current_node, next_node, {})
            edge_type = edge_data.get('type', 'unknown')
            
            # Create walk step
            walk_step = WalkStep(
                node=next_node,
                edge_type=edge_type,
                timestamp=step + 1,
                context=self._get_node_context(next_node, graph)
            )
            walk_path.append(walk_step)
            current_node = next_node
        
        # Generate LLM response
        prompt = self._build_prompt(walk_path, prompt_type)
        llm_response = self._query_llm(prompt)
        
        # Store walk in history
        walk_result = {
            'walk_path': walk_path,
            'prompt': prompt,
            'llm_response': llm_response,
            'metadata': {
                'strategy': self.config.strategy.value,
                'steps': len(walk_path),
                'start_node': start_node,
                'end_node': current_node
            }
        }
        
        self.walk_history.append(walk_result)
        return walk_result
    
    def _get_node_context(self, node: str, graph: nx.Graph) -> Dict[str, Any]:
        """Extract contextual information about a node"""
        node_data = graph.nodes.get(node, {})
        neighbors = list(graph.neighbors(node))
        
        return {
            'label': node_data.get('label', node),
            'type': node_data.get('type', 'unknown'),
            'description': node_data.get('description', ''),
            'neighbors': neighbors[:5],  # Limit for brevity
            'degree': graph.degree(node)
        }
    
    def _build_prompt(self, walk_path: List[WalkStep], prompt_type: str) -> str:
        """Build dynamic prompt based on walk path"""
        template = self.prompt_templates.get(prompt_type, self.prompt_templates['exploration'])
        
        # Format walk path for prompt
        path_str = " → ".join([
            f"{step.node}({step.context.get('type', 'unknown')})"
            for step in walk_path
        ])
        
        # Get current node info
        current_step = walk_path[-1]
        
        # Build context summary
        context_summary = self._summarize_walk_context(walk_path)
        
        return template.format(
            walk_path=path_str,
            current_node=current_step.node,
            context=context_summary
        )
    
    def _summarize_walk_context(self, walk_path: List[WalkStep]) -> str:
        """Create a contextual summary of the walk"""
        if len(walk_path) <= 1:
            return f"Starting exploration at {walk_path[0].node}"
        
        start_node = walk_path[0].node
        end_node = walk_path[-1].node
        
        # Identify entity types traversed
        types_seen = set()
        for step in walk_path:
            node_type = step.context.get('type', 'unknown')
            types_seen.add(node_type)
        
        # Identify relationship types used
        edge_types = set()
        for step in walk_path[1:]:
            if step.edge_type:
                edge_types.add(step.edge_type)
        
        context = f"Journey from {start_node} to {end_node}, "
        context += f"traversing {len(walk_path)} nodes of types: {', '.join(types_seen)}. "
        
        if edge_types:
            context += f"Relationships explored: {', '.join(edge_types)}."
        
        return context
    
    def _query_llm(self, prompt: str) -> str:
        """Query the language model with the generated prompt"""
        # Placeholder for actual LLM integration
        # This would integrate with OpenAI, Hugging Face, or other LLM APIs
        return f"[LLM Response to: {prompt[:100]}...]"
    
    def analyze_walk_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across multiple walks"""
        if not self.walk_history:
            return {}
        
        analysis = {
            'total_walks': len(self.walk_history),
            'avg_steps': np.mean([len(walk['walk_path']) for walk in self.walk_history]),
            'node_frequency': defaultdict(int),
            'edge_frequency': defaultdict(int),
            'strategy_performance': defaultdict(list)
        }
        
        for walk in self.walk_history:
            strategy = walk['metadata']['strategy']
            steps = walk['metadata']['steps']
            analysis['strategy_performance'][strategy].append(steps)
            
            for step in walk['walk_path']:
                analysis['node_frequency'][step.node] += 1
                if step.edge_type:
                    analysis['edge_frequency'][step.edge_type] += 1
        
        return analysis
    
    def get_walk_recommendations(self, current_node: str, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Suggest interesting walk directions based on graph structure and history"""
        neighbors = list(graph.neighbors(current_node))
        
        recommendations = []
        for neighbor in neighbors:
            edge_data = graph.get_edge_data(current_node, neighbor, {})
            
            recommendation = {
                'target_node': neighbor,
                'edge_type': edge_data.get('type', 'unknown'),
                'reason': self._generate_recommendation_reason(current_node, neighbor, graph),
                'estimated_interest': self._calculate_interest_score(neighbor, graph)
            }
            recommendations.append(recommendation)
        
        # Sort by interest score
        recommendations.sort(key=lambda x: x['estimated_interest'], reverse=True)
        return recommendations[:5]  # Return top 5
    
    def _generate_recommendation_reason(self, current_node: str, target_node: str, 
                                      graph: nx.Graph) -> str:
        """Generate explanation for why a particular walk direction is interesting"""
        target_data = graph.nodes.get(target_node, {})
        target_type = target_data.get('type', 'unknown')
        target_degree = graph.degree(target_node)
        
        if target_degree > 10:
            return f"Hub node ({target_type}) with many connections ({target_degree})"
        elif target_degree == 1:
            return f"Leaf node ({target_type}) - potential endpoint for focused exploration"
        else:
            return f"Moderate connectivity ({target_type}) - balanced exploration opportunity"
    
    def _calculate_interest_score(self, node: str, graph: nx.Graph) -> float:
        """Calculate how interesting a node might be for exploration"""
        # Simple heuristic - can be made more sophisticated
        degree = graph.degree(node)
        node_data = graph.nodes.get(node, {})
        
        # Favor nodes with moderate degree (not too isolated, not too common)
        degree_score = 1.0 / (1.0 + abs(degree - 5))
        
        # Bonus for nodes with rich metadata
        metadata_score = len(node_data) * 0.1
        
        # Random factor for exploration
        random_factor = random.random() * 0.3
        
        return degree_score + metadata_score + random_factor

# Example usage and testing
if __name__ == "__main__":
    # Create a sample knowledge graph
    G = nx.Graph()
    
    # Add nodes with metadata
    nodes = [
        ("AI", {"type": "field", "description": "Artificial Intelligence"}),
        ("ML", {"type": "subfield", "description": "Machine Learning"}),
        ("NLP", {"type": "subfield", "description": "Natural Language Processing"}),
        ("Transformers", {"type": "architecture", "description": "Attention-based neural networks"}),
        ("BERT", {"type": "model", "description": "Bidirectional Encoder Representations"}),
        ("GPT", {"type": "model", "description": "Generative Pre-trained Transformer"}),
        ("Knowledge Graphs", {"type": "concept", "description": "Structured knowledge representation"}),
        ("Reasoning", {"type": "capability", "description": "Logical inference and deduction"})
    ]
    
    for node_id, data in nodes:
        G.add_node(node_id, **data)
    
    # Add edges with relationship types
    edges = [
        ("AI", "ML", {"type": "contains"}),
        ("AI", "NLP", {"type": "contains"}),
        ("ML", "Transformers", {"type": "uses"}),
        ("NLP", "Transformers", {"type": "uses"}),
        ("Transformers", "BERT", {"type": "implements"}),
        ("Transformers", "GPT", {"type": "implements"}),
        ("AI", "Knowledge Graphs", {"type": "utilizes"}),
        ("AI", "Reasoning", {"type": "enables"}),
        ("Knowledge Graphs", "Reasoning", {"type": "supports"})
    ]
    
    for src, dst, data in edges:
        G.add_edge(src, dst, **data)
    
    # Initialize enhanced WalkLLM
    config = WalkConfig(
        strategy=WalkStrategy.BIASED,
        max_steps=5,
        temperature=0.8
    )
    
    # Mock LLM client
    class MockLLMClient:
        def query(self, prompt):
            return f"Mock response to prompt: {prompt[:50]}..."
    
    walker = EnhancedWalkLLM(MockLLMClient(), config)
    
    # Perform a walk
    result = walker.perform_walk(G, "AI", "exploration")
    
    print("Walk Result:")
    print(f"Path: {' → '.join([step.node for step in result['walk_path']])}")
    print(f"Steps: {result['metadata']['steps']}")
    print(f"LLM Response: {result['llm_response']}")
    
    # Get recommendations
    recommendations = walker.get_walk_recommendations("AI", G)
    print("\nRecommendations from AI:")
    for rec in recommendations:
        print(f"- {rec['target_node']}: {rec['reason']} (score: {rec['estimated_interest']:.2f})")
