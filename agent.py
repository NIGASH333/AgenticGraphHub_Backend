"""
Retrieval Agent for the RAG system

Decides between graph, vector, and hybrid retrieval strategies.
This is where the "intelligence" happens.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Intelligent retrieval agent for deciding and executing retrieval strategies."""
    
    def __init__(self, openai_api_key: str, graph_builder, vector_store):
        """
        Initialize the RetrievalAgent.
        
        Args:
            openai_api_key: OpenAI API key
            graph_builder: GraphBuilder instance
            vector_store: VectorStore instance
        """
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        self.graph_builder = graph_builder
        self.vector_store = vector_store
        
        # Load prompts
        self.decision_prompt = self._load_prompt("retrieval_decision_prompt.txt")
        
        # System message for reasoning
        self.system_message = SystemMessage(content="""
You are an intelligent retrieval agent that analyzes user queries and provides comprehensive answers
by combining information from knowledge graphs and vector search. Always provide well-structured,
accurate, and helpful responses based on the retrieved information.
        """)
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        prompt_path = Path(__file__).parent / "prompts" / filename
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            return ""
    
    def decide_retrieval_mode(self, query: str) -> str:
        """
        Decide the best retrieval mode for a given query.
        
        Args:
            query: User query
            
        Returns:
            Retrieval mode: 'graph', 'vector', or 'hybrid'
        """
        try:
            # Create decision prompt
            prompt = self.decision_prompt.format(query=query)
            
            # Get decision from LLM
            response = self.llm.invoke([HumanMessage(content=prompt)])
            decision = response.content.strip().upper()
            
            # Map to our internal modes - simple keyword matching
            if "GRAPH" in decision:
                return "graph"
            elif "VECTOR" in decision:
                return "vector"
            else:
                return "hybrid"  # Default to hybrid if unclear
                
        except Exception as e:
            logger.error(f"Error in retrieval decision: {e}")
            return "hybrid"  # Default to hybrid on error
    
    def retrieve_graph_information(self, query: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Retrieve information from the knowledge graph.
        
        Args:
            query: User query
            max_depth: Maximum depth for graph traversal
            
        Returns:
            Dictionary containing graph retrieval results
        """
        try:
            if self.graph_builder.use_neo4j:
                return self._retrieve_from_neo4j(query, max_depth)
            else:
                return self._retrieve_from_networkx(query, max_depth)
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return {"entities": [], "relationships": [], "paths": []}
    
    def _retrieve_from_neo4j(self, query: str, max_depth: int) -> Dict[str, Any]:
        """Retrieve information from Neo4j database."""
        with self.graph_builder.neo4j_driver.session() as session:
            # Extract potential entity names from query (simple approach)
            query_words = query.lower().split()
            
            # Find entities that match query words
            entity_query = """
            MATCH (e:Entity)
            WHERE any(word IN $query_words WHERE toLower(e.name) CONTAINS word)
            RETURN e.name as name, e.type as type, e.attributes as attributes
            LIMIT 10
            """
            
            entities = []
            for record in session.run(entity_query, query_words=query_words):
                entities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "attributes": record["attributes"]
                })
            
            # Find relationships involving these entities
            relationships = []
            if entities:
                entity_names = [e["name"] for e in entities]
                rel_query = """
                MATCH (source:Entity)-[r:RELATION]->(target:Entity)
                WHERE source.name IN $entity_names OR target.name IN $entity_names
                RETURN source.name as source, target.name as target, 
                       r.relation_type as relation, r.attributes as attributes
                LIMIT 20
                """
                
                for record in session.run(rel_query, entity_names=entity_names):
                    relationships.append({
                        "source": record["source"],
                        "target": record["target"],
                        "relation": record["relation"],
                        "attributes": record["attributes"]
                    })
            
            return {
                "entities": entities,
                "relationships": relationships,
                "paths": []  # Could implement path finding here
            }
    
    def _retrieve_from_networkx(self, query: str, max_depth: int) -> Dict[str, Any]:
        """Retrieve information from NetworkX graph."""
        graph = self.graph_builder.graph
        if graph.number_of_nodes() == 0:
            return {"entities": [], "relationships": [], "paths": []}
        
        # Extract potential entity names from query
        query_words = query.lower().split()
        
        # Find matching entities
        entities = []
        for node, data in graph.nodes(data=True):
            if any(word in node.lower() for word in query_words):
                entities.append({
                    "name": node,
                    "type": data.get("type", "UNKNOWN"),
                    "attributes": data.get("attributes", {})
                })
        
        # Find relationships involving these entities
        relationships = []
        entity_names = [e["name"] for e in entities]
        
        for source, target, data in graph.edges(data=True):
            if source in entity_names or target in entity_names:
                relationships.append({
                    "source": source,
                    "target": target,
                    "relation": data.get("relation_type", "UNKNOWN"),
                    "attributes": data.get("attributes", {})
                })
        
        # Find paths between entities (simplified)
        paths = []
        if len(entities) >= 2:
            try:
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        source = entities[i]["name"]
                        target = entities[j]["name"]
                        
                        if nx.has_path(graph, source, target):
                            path = nx.shortest_path(graph, source, target)
                            paths.append({
                                "source": source,
                                "target": target,
                                "path": path
                            })
            except Exception as e:
                logger.error(f"Error finding paths: {e}")
        
        return {
            "entities": entities[:10],  # Limit results
            "relationships": relationships[:20],
            "paths": paths[:5]
        }
    
    def retrieve_vector_information(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieve information from vector store.
        
        Args:
            query: User query
            k: Number of results to return
            
        Returns:
            Dictionary containing vector retrieval results
        """
        try:
            results = self.vector_store.search(query, k)
            return {
                "chunks": results,
                "total_results": len(results)
            }
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            return {"chunks": [], "total_results": 0}
    
    def hybrid_retrieve(self, query: str, graph_k: int = 3, vector_k: int = 3) -> Dict[str, Any]:
        """
        Perform hybrid retrieval combining graph and vector search.
        
        Args:
            query: User query
            graph_k: Number of graph results
            vector_k: Number of vector results
            
        Returns:
            Dictionary containing hybrid retrieval results
        """
        # Get both graph and vector results
        graph_results = self.retrieve_graph_information(query)
        vector_results = self.retrieve_vector_information(query, vector_k)
        
        return {
            "graph": graph_results,
            "vector": vector_results,
            "combined": True
        }
    
    def generate_answer(self, query: str, retrieval_results: Dict[str, Any], 
                       retrieval_mode: str) -> str:
        """
        Generate a comprehensive answer based on retrieval results.
        
        Args:
            query: Original user query
            retrieval_results: Results from retrieval
            retrieval_mode: Mode used for retrieval
            
        Returns:
            Generated answer
        """
        try:
            # Prepare context based on retrieval mode
            context_parts = []
            
            if retrieval_mode == "graph" or (retrieval_mode == "hybrid" and "graph" in retrieval_results):
                graph_data = retrieval_results.get("graph", retrieval_results)
                
                if graph_data.get("entities"):
                    context_parts.append("Relevant entities found:")
                    for entity in graph_data["entities"][:5]:  # Limit to first 5
                        context_parts.append(f"- {entity['name']} ({entity['type']})")
                
                if graph_data.get("relationships"):
                    context_parts.append("\nRelationships found:")
                    for rel in graph_data["relationships"][:5]:  # Limit to first 5
                        context_parts.append(f"- {rel['source']} --[{rel['relation']}]--> {rel['target']}")
            
            if retrieval_mode == "vector" or (retrieval_mode == "hybrid" and "vector" in retrieval_results):
                vector_data = retrieval_results.get("vector", retrieval_results)
                
                if vector_data.get("chunks"):
                    context_parts.append("\nRelevant text passages:")
                    for i, chunk in enumerate(vector_data["chunks"][:3], 1):  # Limit to first 3
                        context_parts.append(f"{i}. {chunk['text'][:200]}...")  # Truncate long text
            
            # Create final prompt
            context = "\n".join(context_parts)
            prompt = f"""
Query: {query}

Context from knowledge base:
{context}

Please provide a comprehensive answer to the query based on the context above.
If the context doesn't contain enough information, please say so.
Be specific and cite relevant entities or relationships when possible.
            """
            
            # Generate answer
            response = self.llm.invoke([
                self.system_message,
                HumanMessage(content=prompt)
            ])
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while processing your query: {str(e)}"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query end-to-end.
        
        Args:
            query: User query
            
        Returns:
            Complete response with answer and metadata
        """
        try:
            # Decide retrieval mode
            retrieval_mode = self.decide_retrieval_mode(query)
            logger.info(f"Selected retrieval mode: {retrieval_mode}")
            
            # Retrieve information
            if retrieval_mode == "graph":
                retrieval_results = self.retrieve_graph_information(query)
            elif retrieval_mode == "vector":
                retrieval_results = self.retrieve_vector_information(query)
            else:  # hybrid
                retrieval_results = self.hybrid_retrieve(query)
            
            # Generate answer
            answer = self.generate_answer(query, retrieval_results, retrieval_mode)
            
            # Prepare response
            response = {
                "query": query,
                "retrieval_mode": retrieval_mode,
                "sources": self._get_sources_used(retrieval_mode, retrieval_results),
                "answer": answer,
                "metadata": {
                    "retrieval_results": retrieval_results
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "retrieval_mode": "error",
                "sources": [],
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def _get_sources_used(self, retrieval_mode: str, retrieval_results: Dict[str, Any]) -> List[str]:
        """Determine which sources were used for retrieval."""
        sources = []
        
        if retrieval_mode == "graph":
            sources.append("graph")
        elif retrieval_mode == "vector":
            sources.append("faiss")
        else:  # hybrid
            if "graph" in retrieval_results and retrieval_results["graph"].get("entities"):
                sources.append("graph")
            if "vector" in retrieval_results and retrieval_results["vector"].get("chunks"):
                sources.append("faiss")
        
        return sources
