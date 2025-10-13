"""
Query Generation Module

Generates Cypher and Gremlin queries from natural language using LLM.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuery:
    """Represents a generated database query."""
    query: str
    query_type: str  # 'cypher' or 'gremlin'
    intent: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]


class QueryGenerator:
    """Generates Cypher and Gremlin queries from natural language."""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the QueryGenerator.
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        # System prompts for different query types
        self.cypher_prompt = self._load_cypher_prompt()
        self.gremlin_prompt = self._load_gremlin_prompt()
    
    def _load_cypher_prompt(self) -> str:
        """Load Cypher generation prompt."""
        return """
You are an expert Neo4j Cypher query generator. Convert natural language queries into Cypher queries.

Guidelines:
1. Use proper Cypher syntax with MATCH, WHERE, RETURN clauses
2. Use parameterized queries when possible
3. Include appropriate indexes and constraints
4. Handle relationships with proper direction (->, <-, -)
5. Use LIMIT for large result sets
6. Include error handling considerations

Common patterns:
- Find entities: MATCH (n:Entity) WHERE n.name CONTAINS $search RETURN n
- Find relationships: MATCH (a)-[r:RELATION]->(b) WHERE r.relation_type = $type RETURN a, r, b
- Path finding: MATCH path = (a)-[*1..5]-(b) WHERE a.name = $start AND b.name = $end RETURN path
- Aggregations: MATCH (n:Entity) RETURN n.type, count(n) as count

Always return a JSON object with:
- "query": the Cypher query string
- "intent": what the query is trying to accomplish
- "confidence": confidence score (0-1)
- "reasoning": explanation of the query logic
- "parameters": any parameters needed
"""
    
    def _load_gremlin_prompt(self) -> str:
        """Load Gremlin generation prompt."""
        return """
You are an expert Apache TinkerPop Gremlin query generator. Convert natural language queries into Gremlin queries.

Guidelines:
1. Use proper Gremlin syntax with g.V(), g.E() traversal methods
2. Chain traversal steps appropriately
3. Use labels, properties, and relationships correctly
4. Include appropriate filtering and limiting
5. Handle both vertex and edge traversals

Common patterns:
- Find vertices: g.V().hasLabel('Entity').has('name', textContains($search))
- Find edges: g.E().hasLabel('RELATION').has('relation_type', $type)
- Path finding: g.V().has('name', $start).repeat(out()).until(has('name', $end)).path()
- Aggregations: g.V().hasLabel('Entity').groupCount().by('type')

Always return a JSON object with:
- "query": the Gremlin query string
- "intent": what the query is trying to accomplish
- "confidence": confidence score (0-1)
- "reasoning": explanation of the query logic
- "parameters": any parameters needed
"""
    
    def generate_cypher_query(self, natural_query: str, 
                            schema_info: Optional[Dict] = None) -> GeneratedQuery:
        """
        Generate a Cypher query from natural language.
        
        Args:
            natural_query: Natural language query
            schema_info: Optional schema information for context
            
        Returns:
            GeneratedQuery object
        """
        try:
            # Prepare context
            context = ""
            if schema_info:
                context = f"""
Schema Information:
- Node labels: {schema_info.get('node_labels', [])}
- Relationship types: {schema_info.get('relationship_types', [])}
- Common properties: {schema_info.get('properties', [])}
"""
            
            prompt = f"""
{self.cypher_prompt}

{context}

Natural Language Query: {natural_query}

Generate a Cypher query for this request.
"""
            
            response = self.llm.invoke([
                SystemMessage(content=self.cypher_prompt),
                HumanMessage(content=prompt)
            ])
            
            result = json.loads(response.content)
            
            return GeneratedQuery(
                query=result['query'],
                query_type='cypher',
                intent=result['intent'],
                confidence=result['confidence'],
                reasoning=result['reasoning'],
                parameters=result.get('parameters', {})
            )
            
        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            return GeneratedQuery(
                query="MATCH (n) RETURN n LIMIT 10",
                query_type='cypher',
                intent="Error fallback",
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                parameters={}
            )
    
    def generate_gremlin_query(self, natural_query: str, 
                             schema_info: Optional[Dict] = None) -> GeneratedQuery:
        """
        Generate a Gremlin query from natural language.
        
        Args:
            natural_query: Natural language query
            schema_info: Optional schema information for context
            
        Returns:
            GeneratedQuery object
        """
        try:
            # Prepare context
            context = ""
            if schema_info:
                context = f"""
Schema Information:
- Vertex labels: {schema_info.get('vertex_labels', [])}
- Edge labels: {schema_info.get('edge_labels', [])}
- Common properties: {schema_info.get('properties', [])}
"""
            
            prompt = f"""
{self.gremlin_prompt}

{context}

Natural Language Query: {natural_query}

Generate a Gremlin query for this request.
"""
            
            response = self.llm.invoke([
                SystemMessage(content=self.gremlin_prompt),
                HumanMessage(content=prompt)
            ])
            
            result = json.loads(response.content)
            
            return GeneratedQuery(
                query=result['query'],
                query_type='gremlin',
                intent=result['intent'],
                confidence=result['confidence'],
                reasoning=result['reasoning'],
                parameters=result.get('parameters', {})
            )
            
        except Exception as e:
            logger.error(f"Error generating Gremlin query: {e}")
            return GeneratedQuery(
                query="g.V().limit(10)",
                query_type='gremlin',
                intent="Error fallback",
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                parameters={}
            )
    
    def generate_hybrid_query(self, natural_query: str, 
                            schema_info: Optional[Dict] = None) -> Dict[str, GeneratedQuery]:
        """
        Generate both Cypher and Gremlin queries for comparison.
        
        Args:
            natural_query: Natural language query
            schema_info: Optional schema information for context
            
        Returns:
            Dictionary with 'cypher' and 'gremlin' keys
        """
        cypher_query = self.generate_cypher_query(natural_query, schema_info)
        gremlin_query = self.generate_gremlin_query(natural_query, schema_info)
        
        return {
            'cypher': cypher_query,
            'gremlin': gremlin_query
        }
    
    def analyze_query_complexity(self, query: str, query_type: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a generated query.
        
        Args:
            query: The generated query string
            query_type: 'cypher' or 'gremlin'
            
        Returns:
            Complexity analysis dictionary
        """
        analysis = {
            'query_length': len(query),
            'complexity_score': 0.0,
            'features': [],
            'performance_notes': []
        }
        
        if query_type == 'cypher':
            # Analyze Cypher-specific features
            if 'MATCH' in query:
                analysis['features'].append('pattern_matching')
                analysis['complexity_score'] += 0.2
            
            if 'WHERE' in query:
                analysis['features'].append('filtering')
                analysis['complexity_score'] += 0.1
            
            if 'RETURN' in query:
                analysis['features'].append('projection')
                analysis['complexity_score'] += 0.1
            
            if '[*' in query:
                analysis['features'].append('variable_length_paths')
                analysis['complexity_score'] += 0.3
                analysis['performance_notes'].append('Variable length paths can be expensive')
            
            if 'ORDER BY' in query:
                analysis['features'].append('sorting')
                analysis['complexity_score'] += 0.1
            
            if 'LIMIT' in query:
                analysis['features'].append('limiting')
                analysis['complexity_score'] += 0.05
            
            # Count relationship patterns
            relationship_count = len(re.findall(r'\[.*?\]', query))
            if relationship_count > 3:
                analysis['performance_notes'].append('Multiple relationships may impact performance')
        
        elif query_type == 'gremlin':
            # Analyze Gremlin-specific features
            if 'repeat(' in query:
                analysis['features'].append('recursive_traversal')
                analysis['complexity_score'] += 0.3
                analysis['performance_notes'].append('Recursive traversals can be expensive')
            
            if 'groupCount()' in query or 'group()' in query:
                analysis['features'].append('aggregation')
                analysis['complexity_score'] += 0.2
            
            if 'path()' in query:
                analysis['features'].append('path_tracking')
                analysis['complexity_score'] += 0.2
            
            if 'has(' in query:
                analysis['features'].append('property_filtering')
                analysis['complexity_score'] += 0.1
            
            if 'out(' in query or 'in(' in query:
                analysis['features'].append('edge_traversal')
                analysis['complexity_score'] += 0.1
        
        # Normalize complexity score
        analysis['complexity_score'] = min(analysis['complexity_score'], 1.0)
        
        return analysis
    
    def optimize_query(self, query: str, query_type: str, 
                     performance_hints: List[str] = None) -> str:
        """
        Optimize a generated query for better performance.
        
        Args:
            query: The query to optimize
            query_type: 'cypher' or 'gremlin'
            performance_hints: Optional performance hints
            
        Returns:
            Optimized query string
        """
        optimized = query
        
        if query_type == 'cypher':
            # Add LIMIT if not present and query might return many results
            if 'LIMIT' not in query and ('MATCH' in query or 'RETURN' in query):
                optimized += ' LIMIT 100'
            
            # Add index hints for common patterns
            if 'WHERE' in query and 'CONTAINS' in query:
                # Suggest text index usage
                optimized = optimized.replace(
                    'WHERE n.name CONTAINS',
                    'WHERE n.name CONTAINS'  # Could add index hint here
                )
        
        elif query_type == 'gremlin':
            # Add limit if not present
            if 'limit(' not in query.lower():
                optimized += '.limit(100)'
            
            # Optimize traversal order
            if '.has(' in optimized and '.out(' in optimized:
                # Move has() before out() for better performance
                parts = optimized.split('.')
                has_parts = [p for p in parts if p.startswith('has(')]
                other_parts = [p for p in parts if not p.startswith('has(')]
                optimized = '.'.join(has_parts + other_parts)
        
        return optimized
    
    def validate_query_syntax(self, query: str, query_type: str) -> Dict[str, Any]:
        """
        Validate query syntax and provide feedback.
        
        Args:
            query: The query to validate
            query_type: 'cypher' or 'gremlin'
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        if query_type == 'cypher':
            # Basic Cypher syntax checks
            if not query.strip().upper().startswith(('MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE')):
                validation['warnings'].append('Query should start with a valid Cypher clause')
            
            if 'MATCH' in query and 'RETURN' not in query:
                validation['warnings'].append('MATCH queries should include RETURN clause')
            
            if query.count('(') != query.count(')'):
                validation['errors'].append('Mismatched parentheses')
                validation['is_valid'] = False
            
            if query.count('[') != query.count(']'):
                validation['errors'].append('Mismatched square brackets')
                validation['is_valid'] = False
        
        elif query_type == 'gremlin':
            # Basic Gremlin syntax checks
            if not query.strip().startswith('g.'):
                validation['warnings'].append('Gremlin queries should start with g.')
            
            if query.count('(') != query.count(')'):
                validation['errors'].append('Mismatched parentheses')
                validation['is_valid'] = False
        
        return validation
