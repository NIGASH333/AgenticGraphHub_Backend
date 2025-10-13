"""
Entity Deduplication Module

Advanced algorithms for identifying and merging duplicate entities in knowledge graphs.
"""

import logging
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import networkx as nx
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import json

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """Represents a potential match between two entities."""
    entity1: str
    entity2: str
    similarity_score: float
    match_type: str  # 'exact', 'fuzzy', 'semantic', 'contextual'
    confidence: float
    reasoning: str


class EntityDeduplicator:
    """Advanced entity deduplication with multiple matching strategies."""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the EntityDeduplicator.
        
        Args:
            openai_api_key: OpenAI API key for semantic matching
        """
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        # Similarity thresholds
        self.exact_threshold = 1.0
        self.fuzzy_threshold = 0.85
        self.semantic_threshold = 0.75
        self.contextual_threshold = 0.65
    
    def find_duplicates(self, entities: List[Dict[str, Any]]) -> List[EntityMatch]:
        """
        Find potential duplicate entities using multiple strategies.
        
        Args:
            entities: List of entity dictionaries with name, type, attributes
            
        Returns:
            List of EntityMatch objects representing potential duplicates
        """
        matches = []
        
        # Group entities by type for more efficient comparison
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get('type', 'UNKNOWN')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Find matches within each type group
        for entity_type, type_entities in entities_by_type.items():
            logger.info(f"Processing {len(type_entities)} entities of type {entity_type}")
            
            for i, entity1 in enumerate(type_entities):
                for j, entity2 in enumerate(type_entities[i+1:], i+1):
                    match = self._compare_entities(entity1, entity2)
                    if match:
                        matches.append(match)
        
        # Sort by confidence score
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Found {len(matches)} potential duplicate pairs")
        return matches
    
    def _compare_entities(self, entity1: Dict, entity2: Dict) -> EntityMatch:
        """
        Compare two entities and determine if they are duplicates.
        
        Args:
            entity1: First entity dictionary
            entity2: Second entity dictionary
            
        Returns:
            EntityMatch if entities are potential duplicates, None otherwise
        """
        name1 = entity1.get('name', '').lower().strip()
        name2 = entity2.get('name', '').lower().strip()
        
        if name1 == name2:
            return EntityMatch(
                entity1=entity1['name'],
                entity2=entity2['name'],
                similarity_score=1.0,
                match_type='exact',
                confidence=1.0,
                reasoning="Exact name match"
            )
        
        # Calculate string similarity
        string_similarity = SequenceMatcher(None, name1, name2).ratio()
        
        if string_similarity >= self.fuzzy_threshold:
            return EntityMatch(
                entity1=entity1['name'],
                entity2=entity2['name'],
                similarity_score=string_similarity,
                match_type='fuzzy',
                confidence=string_similarity,
                reasoning=f"String similarity: {string_similarity:.2f}"
            )
        
        # Check for semantic similarity using LLM
        semantic_score = self._calculate_semantic_similarity(entity1, entity2)
        if semantic_score >= self.semantic_threshold:
            return EntityMatch(
                entity1=entity1['name'],
                entity2=entity2['name'],
                similarity_score=semantic_score,
                match_type='semantic',
                confidence=semantic_score,
                reasoning=f"Semantic similarity: {semantic_score:.2f}"
            )
        
        # Check contextual similarity (attributes, relationships)
        contextual_score = self._calculate_contextual_similarity(entity1, entity2)
        if contextual_score >= self.contextual_threshold:
            return EntityMatch(
                entity1=entity1['name'],
                entity2=entity2['name'],
                similarity_score=contextual_score,
                match_type='contextual',
                confidence=contextual_score,
                reasoning=f"Contextual similarity: {contextual_score:.2f}"
            )
        
        return None
    
    def _calculate_semantic_similarity(self, entity1: Dict, entity2: Dict) -> float:
        """
        Calculate semantic similarity using LLM.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            prompt = f"""
            Compare these two entities and determine if they refer to the same real-world entity.
            
            Entity 1: {entity1['name']} (Type: {entity1.get('type', 'UNKNOWN')})
            Attributes: {json.dumps(entity1.get('attributes', {}))}
            
            Entity 2: {entity2['name']} (Type: {entity2.get('type', 'UNKNOWN')})
            Attributes: {json.dumps(entity2.get('attributes', {}))}
            
            Respond with a JSON object containing:
            - "similarity": float between 0 and 1 (1 = definitely same entity)
            - "reasoning": brief explanation
            
            Consider:
            - Name variations (nicknames, abbreviations, different spellings)
            - Type consistency
            - Attribute overlap
            - Real-world entity identity
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            return result.get('similarity', 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_contextual_similarity(self, entity1: Dict, entity2: Dict) -> float:
        """
        Calculate contextual similarity based on attributes and relationships.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Similarity score between 0 and 1
        """
        score = 0.0
        total_weight = 0.0
        
        # Compare attributes
        attrs1 = entity1.get('attributes', {})
        attrs2 = entity2.get('attributes', {})
        
        if attrs1 and attrs2:
            common_keys = set(attrs1.keys()) & set(attrs2.keys())
            if common_keys:
                attr_similarity = 0.0
                for key in common_keys:
                    val1 = str(attrs1[key]).lower()
                    val2 = str(attrs2[key]).lower()
                    if val1 == val2:
                        attr_similarity += 1.0
                    else:
                        # Use string similarity for non-exact matches
                        attr_similarity += SequenceMatcher(None, val1, val2).ratio()
                
                attr_similarity /= len(common_keys)
                score += attr_similarity * 0.4  # 40% weight for attributes
                total_weight += 0.4
        
        # Type consistency
        type1 = entity1.get('type', 'UNKNOWN')
        type2 = entity2.get('type', 'UNKNOWN')
        if type1 == type2:
            score += 0.3  # 30% weight for type consistency
        total_weight += 0.3
        
        # Name similarity (already calculated)
        name_similarity = SequenceMatcher(
            None, 
            entity1.get('name', '').lower(), 
            entity2.get('name', '').lower()
        ).ratio()
        score += name_similarity * 0.3  # 30% weight for name
        total_weight += 0.3
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def merge_entities(self, entity_matches: List[EntityMatch], 
                      entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge duplicate entities based on match results.
        
        Args:
            entity_matches: List of confirmed duplicate matches
            entities: Original list of entities
            
        Returns:
            List of merged entities
        """
        # Create entity lookup
        entity_lookup = {entity['name']: entity for entity in entities}
        merged_entities = []
        processed_entities = set()
        
        for match in entity_matches:
            if match.entity1 in processed_entities or match.entity2 in processed_entities:
                continue
            
            entity1 = entity_lookup.get(match.entity1)
            entity2 = entity_lookup.get(match.entity2)
            
            if not entity1 or not entity2:
                continue
            
            # Merge entities
            merged_entity = self._merge_two_entities(entity1, entity2, match)
            merged_entities.append(merged_entity)
            processed_entities.add(match.entity1)
            processed_entities.add(match.entity2)
        
        # Add non-merged entities
        for entity in entities:
            if entity['name'] not in processed_entities:
                merged_entities.append(entity)
        
        logger.info(f"Merged {len(entity_matches)} duplicate pairs, "
                   f"resulting in {len(merged_entities)} unique entities")
        
        return merged_entities
    
    def _merge_two_entities(self, entity1: Dict, entity2: Dict, match: EntityMatch) -> Dict:
        """
        Merge two entities into one.
        
        Args:
            entity1: First entity
            entity2: Second entity
            match: Match information
            
        Returns:
            Merged entity
        """
        # Use the entity with more attributes as base
        if len(entity1.get('attributes', {})) >= len(entity2.get('attributes', {})):
            base_entity = entity1.copy()
            other_entity = entity2
        else:
            base_entity = entity2.copy()
            other_entity = entity1
        
        # Merge attributes
        merged_attributes = base_entity.get('attributes', {}).copy()
        for key, value in other_entity.get('attributes', {}).items():
            if key not in merged_attributes:
                merged_attributes[key] = value
            elif merged_attributes[key] != value:
                # Handle conflicting values
                if isinstance(merged_attributes[key], list):
                    if value not in merged_attributes[key]:
                        merged_attributes[key].append(value)
                else:
                    merged_attributes[key] = [merged_attributes[key], value]
        
        # Create merged entity
        merged_entity = {
            'name': base_entity['name'],
            'type': base_entity.get('type', 'UNKNOWN'),
            'attributes': merged_attributes,
            'merged_from': [entity1['name'], entity2['name']],
            'merge_reason': match.reasoning,
            'merge_confidence': match.confidence
        }
        
        return merged_entity
    
    def deduplicate_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deduplicate entities in a complete graph structure.
        
        Args:
            graph_data: Dictionary with 'nodes' and 'edges' keys
            
        Returns:
            Deduplicated graph data
        """
        logger.info("Starting graph deduplication...")
        
        # Find duplicates
        matches = self.find_duplicates(graph_data['nodes'])
        
        # Filter high-confidence matches
        high_confidence_matches = [
            match for match in matches 
            if match.confidence >= self.semantic_threshold
        ]
        
        logger.info(f"Found {len(high_confidence_matches)} high-confidence duplicates")
        
        # Merge entities
        deduplicated_nodes = self.merge_entities(
            high_confidence_matches, 
            graph_data['nodes']
        )
        
        # Update edges to reference merged entities
        deduplicated_edges = self._update_edges_for_merged_entities(
            graph_data['edges'], 
            high_confidence_matches
        )
        
        return {
            'nodes': deduplicated_nodes,
            'edges': deduplicated_edges,
            'deduplication_stats': {
                'original_entities': len(graph_data['nodes']),
                'deduplicated_entities': len(deduplicated_nodes),
                'duplicates_found': len(high_confidence_matches),
                'reduction_percentage': (
                    (len(graph_data['nodes']) - len(deduplicated_nodes)) / 
                    len(graph_data['nodes']) * 100
                )
            }
        }
    
    def _update_edges_for_merged_entities(self, edges: List[Dict], 
                                        matches: List[EntityMatch]) -> List[Dict]:
        """
        Update edge references after entity merging.
        
        Args:
            edges: List of edge dictionaries
            matches: List of entity matches that were merged
            
        Returns:
            Updated edges
        """
        # Create mapping from old names to new names
        name_mapping = {}
        for match in matches:
            # Use the first entity name as the canonical name
            name_mapping[match.entity2] = match.entity1
        
        updated_edges = []
        for edge in edges:
            source = name_mapping.get(edge['source'], edge['source'])
            target = name_mapping.get(edge['target'], edge['target'])
            
            # Skip self-loops
            if source != target:
                updated_edges.append({
                    'source': source,
                    'target': target,
                    'relation': edge['relation'],
                    'attributes': edge.get('attributes', {})
                })
        
        return updated_edges
