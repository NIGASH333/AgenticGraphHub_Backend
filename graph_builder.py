"""
Graph Builder for the RAG system

Handles document processing and knowledge graph construction.
TODO: Add support for more file types (PDF, Word docs)
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import networkx as nx
from neo4j import GraphDatabase
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphBuilder:
    """Handles document processing and knowledge graph construction."""
    
    def __init__(self, openai_api_key: str, neo4j_uri: Optional[str] = None, 
                 neo4j_username: Optional[str] = None, neo4j_password: Optional[str] = None):
        """
        Initialize the GraphBuilder.
        
        Args:
            openai_api_key: OpenAI API key for LLM operations
            neo4j_uri: Neo4j database URI (optional)
            neo4j_username: Neo4j username (optional)
            neo4j_password: Neo4j password (optional)
        """
        self.openai_api_key = openai_api_key
        
        # Initialize LLM - using 3.5 turbo for cost reasons
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1  # Low temperature for more consistent extraction
        )
        
        # We don't actually use embeddings here but keeping for consistency
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Try to connect to Neo4j, fallback to NetworkX
        self.use_neo4j = all([neo4j_uri, neo4j_username, neo4j_password])
        if self.use_neo4j:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    neo4j_uri, 
                    auth=(neo4j_username, neo4j_password)
                )
                logger.info("Connected to Neo4j database")
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}. Using NetworkX instead.")
                self.use_neo4j = False
        
        if not self.use_neo4j:
            self.graph = nx.MultiDiGraph()
            logger.info("Using NetworkX for graph storage")
        
        # Load the extraction prompt
        self.extract_prompt = self._load_prompt("extract_graph_prompt.txt")
        
        # Text splitter - same settings as vector store
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        prompt_path = Path(__file__).parent / "prompts" / filename
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            return ""
    
    def process_documents(self, data_dir: str) -> Dict[str, Any]:
        """
        Process all documents in the data directory.
        
        Args:
            data_dir: Path to directory containing documents
            
        Returns:
            Dictionary containing processing results
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return {"error": "Data directory not found"}
        
        results = {
            "processed_files": [],
            "total_chunks": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "errors": []
        }
        
        # Process all supported file types
        supported_extensions = ['.txt', '.md', '.py']
        for file_path in data_path.glob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    file_result = self._process_single_document(file_path)
                    results["processed_files"].append(file_path.name)
                    results["total_chunks"] += file_result.get("chunks", 0)
                    results["total_entities"] += file_result.get("entities", 0)
                    results["total_relationships"] += file_result.get("relationships", 0)
                except Exception as e:
                    error_msg = f"Error processing {file_path.name}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
        
        # Save graph data
        self._save_graph_data()
        
        logger.info(f"Processing complete. Processed {len(results['processed_files'])} files.")
        return results
    
    def _process_single_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing processing results for this file
        """
        logger.info(f"Processing file: {file_path.name}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        
        file_result = {
            "chunks": len(chunks),
            "entities": 0,
            "relationships": 0
        }
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                chunk_result = self._process_chunk(chunk, f"{file_path.stem}_chunk_{i}")
                file_result["entities"] += chunk_result.get("entities", 0)
                file_result["relationships"] += chunk_result.get("relationships", 0)
            except Exception as e:
                logger.error(f"Error processing chunk {i} in {file_path.name}: {str(e)}")
        
        return file_result
    
    def _process_chunk(self, chunk: str, chunk_id: str) -> Dict[str, Any]:
        """
        Process a single text chunk to extract triples.
        
        Args:
            chunk: Text content to process
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Dictionary containing extraction results
        """
        # Create extraction prompt
        prompt = f"{self.extract_prompt}\n\n{chunk}"
        
        try:
            # Extract triples using LLM - this is where the magic happens
            response = self.llm.invoke([HumanMessage(content=prompt)])
            extraction_result = json.loads(response.content)
            
            # Store in graph database
            entities_added = self._store_entities(extraction_result.get("entities", []), chunk_id)
            relationships_added = self._store_relationships(extraction_result.get("relationships", []), chunk_id)
            
            return {
                "entities": entities_added,
                "relationships": relationships_added
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            # Sometimes the LLM doesn't return valid JSON, just skip this chunk
            return {"entities": 0, "relationships": 0}
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            return {"entities": 0, "relationships": 0}
    
    def _store_entities(self, entities: List[Dict], chunk_id: str) -> int:
        """Store entities in the graph database."""
        entities_added = 0
        
        for entity in entities:
            try:
                if self.use_neo4j:
                    self._store_entity_neo4j(entity, chunk_id)
                else:
                    self._store_entity_networkx(entity, chunk_id)
                entities_added += 1
            except Exception as e:
                logger.error(f"Error storing entity {entity.get('name', 'unknown')}: {e}")
        
        return entities_added
    
    def _store_relationships(self, relationships: List[Dict], chunk_id: str) -> int:
        """Store relationships in the graph database."""
        relationships_added = 0
        
        for rel in relationships:
            try:
                if self.use_neo4j:
                    self._store_relationship_neo4j(rel, chunk_id)
                else:
                    self._store_relationship_networkx(rel, chunk_id)
                relationships_added += 1
            except Exception as e:
                logger.error(f"Error storing relationship {rel.get('relation', 'unknown')}: {e}")
        
        return relationships_added
    
    def _store_entity_neo4j(self, entity: Dict, chunk_id: str):
        """Store entity in Neo4j."""
        # Flatten attributes to individual properties
        attributes = entity.get("attributes", {})
        
        # Build the SET clause dynamically for each attribute
        set_clauses = ["e.type = $type", "e.chunk_id = $chunk_id"]
        params = {
            "name": entity["name"],
            "type": entity["type"],
            "chunk_id": chunk_id
        }
        
        # Add each attribute as a separate property
        for key, value in attributes.items():
            # Sanitize key name for Neo4j (remove spaces, special chars)
            sanitized_key = key.replace(" ", "_").replace("-", "_")
            
            # Convert complex types to JSON strings
            if isinstance(value, (dict, list)):
                params[f"attr_{sanitized_key}"] = json.dumps(value)
            else:
                params[f"attr_{sanitized_key}"] = value
            set_clauses.append(f"e.{sanitized_key} = $attr_{sanitized_key}")
        
        # Build complete query
        query = f"""
            MERGE (e:Entity {{name: $name}})
            SET {', '.join(set_clauses)}
        """
        
        with self.neo4j_driver.session() as session:
            session.run(query, **params)
    
    def _store_entity_networkx(self, entity: Dict, chunk_id: str):
        """Store entity in NetworkX."""
        self.graph.add_node(
            entity["name"],
            type=entity["type"],
            chunk_id=chunk_id,
            attributes=entity.get("attributes", {})
        )
    
    def _store_relationship_neo4j(self, relationship: Dict, chunk_id: str):
        """Store relationship in Neo4j."""
        # Flatten attributes to individual properties
        attributes = relationship.get("attributes", {})
        
        # Build the SET clause dynamically for each attribute
        set_clauses = ["r.chunk_id = $chunk_id"]
        params = {
            "source": relationship["source"],
            "target": relationship["target"],
            "relation_type": relationship["relation"],
            "chunk_id": chunk_id
        }
        
        # Add each attribute as a separate property
        for key, value in attributes.items():
            # Sanitize key name for Neo4j (remove spaces, special chars)
            sanitized_key = key.replace(" ", "_").replace("-", "_")
            
            # Convert complex types to JSON strings
            if isinstance(value, (dict, list)):
                params[f"attr_{sanitized_key}"] = json.dumps(value)
            else:
                params[f"attr_{sanitized_key}"] = value
            set_clauses.append(f"r.{sanitized_key} = $attr_{sanitized_key}")
        
        # Build complete query
        query = f"""
            MATCH (source:Entity {{name: $source}})
            MATCH (target:Entity {{name: $target}})
            MERGE (source)-[r:RELATION {{relation_type: $relation_type}}]->(target)
            SET {', '.join(set_clauses)}
        """
        
        with self.neo4j_driver.session() as session:
            session.run(query, **params)
    
    def _store_relationship_networkx(self, relationship: Dict, chunk_id: str):
        """Store relationship in NetworkX."""
        self.graph.add_edge(
            relationship["source"],
            relationship["target"],
            relation_type=relationship["relation"],
            chunk_id=chunk_id,
            attributes=relationship.get("attributes", {})
        )
    
    def _save_graph_data(self):
        """Save graph data to JSON file."""
        if not self.use_neo4j:
            # Convert NetworkX graph to JSON format
            graph_data = {
                "nodes": [
                    {
                        "id": node,
                        "type": data.get("type", "UNKNOWN"),
                        "attributes": data.get("attributes", {})
                    }
                    for node, data in self.graph.nodes(data=True)
                ],
                "edges": [
                    {
                        "source": edge[0],
                        "target": edge[1],
                        "relation": data.get("relation_type", "UNKNOWN"),
                        "attributes": data.get("attributes", {})
                    }
                    for edge, data in self.graph.edges(data=True)
                ]
            }
            
            # Save to file
            data_dir = Path(__file__).parent / "data"
            with open(data_dir / "graph_data.json", "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Graph data saved with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        if self.use_neo4j:
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()["node_count"]
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as edge_count")
                edge_count = result.single()["edge_count"]
                
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "database": "Neo4j"
                }
        else:
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "database": "NetworkX"
            }
    
    def close(self):
        """Close database connections."""
        if self.use_neo4j and hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()