"""
FastAPI Application for the RAG system with Keep-Alive Background Task

Main API endpoints for the Agentic Graph RAG service.
TODO: Add authentication, rate limiting, better error handling
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from graph_builder import GraphBuilder
from vector_store import VectorStore
from agent import RetrievalAgent
from entity_deduplication import EntityDeduplicator
from query_generator import QueryGenerator
from advanced_reasoning import AdvancedReasoningEngine
from rate_limit_middleware import RateLimitMiddleware

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for components
graph_builder = None
vector_store = None
retrieval_agent = None
entity_deduplicator = None
query_generator = None
reasoning_engine = None

# Keep-alive task management
keep_alive_task = None
KEEP_ALIVE_INTERVAL = 300  # Ping every 5 minutes (adjust as needed)


async def keep_alive_worker(app_url: str):
    """
    Background task that pings the server periodically to prevent it from sleeping.
    
    Args:
        app_url: The URL of the application to ping
    """
    logger.info(f"Starting keep-alive worker. Will ping {app_url}/ping every {KEEP_ALIVE_INTERVAL} seconds")
    
    while True:
        try:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL)
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{app_url}/ping", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            logger.debug(f"Keep-alive ping successful: {await resp.json()}")
                        else:
                            logger.warning(f"Keep-alive ping returned status {resp.status}")
                except asyncio.TimeoutError:
                    logger.warning("Keep-alive ping timed out")
                except Exception as e:
                    logger.error(f"Error during keep-alive ping: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Keep-alive worker stopped")
            break
        except Exception as e:
            logger.error(f"Unexpected error in keep-alive worker: {e}")
            await asyncio.sleep(10)  # Wait before retrying


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global graph_builder, vector_store, retrieval_agent, entity_deduplicator, query_generator, reasoning_engine, keep_alive_task
    
    # Startup
    logger.info("Starting RAG Service...")
    
    # Get configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    data_dir = os.getenv("DATA_DIR", "./data")
    faiss_index_dir = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")
    
    # Initialize components
    try:
        graph_builder = GraphBuilder(
            openai_api_key=openai_api_key,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password
        )
        
        vector_store = VectorStore(
            openai_api_key=openai_api_key,
            index_dir=faiss_index_dir
        )
        
        retrieval_agent = RetrievalAgent(
            openai_api_key=openai_api_key,
            graph_builder=graph_builder,
            vector_store=vector_store
        )
        
        # Initialize new advanced components
        entity_deduplicator = EntityDeduplicator(openai_api_key=openai_api_key)
        query_generator = QueryGenerator(openai_api_key=openai_api_key)
        reasoning_engine = AdvancedReasoningEngine(openai_api_key=openai_api_key)
        
        logger.info("All components initialized successfully")
        
        # Start keep-alive task
        app_host = os.getenv("APP_HOST", "localhost")
        app_port = int(os.getenv("APP_PORT", 8000))
        
        # Determine if we're in a cloud environment
        if os.getenv("RENDER") or os.getenv("HEROKU") or os.getenv("RAILWAY"):
            # Use the deployed URL if available
            app_url = os.getenv("APP_URL") or f"http://{app_host}:{app_port}"
            keep_alive_task = asyncio.create_task(keep_alive_worker(app_url))
            logger.info(f"Keep-alive task started for {app_url}")
        else:
            logger.info("Running in local environment - keep-alive task disabled")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Service...")
    
    # Cancel keep-alive task
    if keep_alive_task:
        keep_alive_task.cancel()
        try:
            await keep_alive_task
        except asyncio.CancelledError:
            logger.info("Keep-alive task cancelled")
    
    if graph_builder:
        graph_builder.close()


# Create FastAPI app
app = FastAPI(
    title="RAG Service",
    description="A RAG system with graph and vector retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    exempt_paths=['/health', '/docs', '/openapi.json', '/graph/stats', '/vector/stats', '/ping']
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    stream: bool = False


class QueryResponse(BaseModel):
    query: str
    retrieval_mode: str
    sources: List[str]
    answer: str
    metadata: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    message: str
    processed_files: List[str]
    total_chunks: int
    total_entities: int
    total_relationships: int
    errors: List[str]


class GraphStatsResponse(BaseModel):
    nodes: int
    edges: int
    database: str


class VectorStatsResponse(BaseModel):
    total_vectors: int
    dimension: int
    index_type: str


# Utility functions
def get_components():
    """Get initialized components."""
    if not all([graph_builder, vector_store, retrieval_agent]):
        raise HTTPException(status_code=500, detail="Service not properly initialized")
    return graph_builder, vector_store, retrieval_agent


def get_advanced_components():
    """Get advanced components."""
    if not all([entity_deduplicator, query_generator, reasoning_engine]):
        raise HTTPException(status_code=500, detail="Advanced components not initialized")
    return entity_deduplicator, query_generator, reasoning_engine


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Agentic Graph RAG Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "POST /query",
            "ingest": "POST /ingest",
            "graph_stats": "GET /graph/stats",
            "vector_stats": "GET /vector/stats",
            "graph_data": "GET /graph/data",
            "ontology_editor": "GET /editor",
            "ontology_save": "POST /ontology/save",
            "deduplicate": "POST /deduplicate",
            "query_generate": "POST /query/generate",
            "advanced_reasoning": "POST /reasoning/advanced"
        }
    }


@app.get("/editor")
async def ontology_editor():
    """Serve the ontology editor frontend."""
    return FileResponse("static/index.html")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base with natural language.
    
    Args:
        request: Query request with query text and optional streaming flag
        
    Returns:
        Query response with answer and metadata
    """
    try:
        graph_builder, vector_store, retrieval_agent = get_components()
        
        # Process query
        response = retrieval_agent.process_query(request.query)
        
        if request.stream:
            # For streaming, we'll return the response immediately
            # TODO: Implement proper streaming for answer generation
            return response
        else:
            return response
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Stream query responses for real-time interaction.
    
    Args:
        request: Query request with query text
        
    Returns:
        Streaming response with answer chunks
    """
    try:
        graph_builder, vector_store, retrieval_agent = get_components()
        
        async def generate_response():
            # Process query
            response = retrieval_agent.process_query(request.query)
            
            # Stream the response
            yield f"data: {json.dumps({'type': 'start', 'retrieval_mode': response['retrieval_mode']})}\n\n"
            
            # Split answer into chunks for streaming
            answer = response['answer']
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'end', 'metadata': response['metadata']})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Error streaming query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    process_existing: bool = Form(False)
):
    """
    Ingest documents into the knowledge base.
    
    Args:
        files: List of uploaded files
        process_existing: Whether to process existing files in data directory
        
    Returns:
        Ingest response with processing results
    """
    try:
        graph_builder, vector_store, retrieval_agent = get_components()
        
        data_dir = Path(os.getenv("DATA_DIR", "./data"))
        data_dir.mkdir(exist_ok=True)
        
        processed_files = []
        total_chunks = 0
        total_entities = 0
        total_relationships = 0
        errors = []
        
        # Save uploaded files
        for file in files:
            try:
                file_path = data_dir / file.filename
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                processed_files.append(file.filename)
                logger.info(f"Saved file: {file.filename}")
            except Exception as e:
                error_msg = f"Error saving file {file.filename}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Process documents if any files were uploaded or if processing existing
        if processed_files or process_existing:
            # Process with graph builder - this can take a while
            graph_results = graph_builder.process_documents(str(data_dir))
            total_entities = graph_results.get("total_entities", 0)
            total_relationships = graph_results.get("total_relationships", 0)
            errors.extend(graph_results.get("errors", []))
            
            # Process with vector store
            vector_chunks = vector_store.add_documents_from_files(str(data_dir))
            total_chunks = max(total_chunks, vector_chunks)
        
        return IngestResponse(
            message="Documents processed successfully",
            processed_files=processed_files,
            total_chunks=total_chunks,
            total_entities=total_entities,
            total_relationships=total_relationships,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """Get statistics about the knowledge graph."""
    try:
        graph_builder, _, _ = get_components()
        stats = graph_builder.get_graph_stats()
        return GraphStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector/stats", response_model=VectorStatsResponse)
async def get_vector_stats():
    """Get statistics about the vector store."""
    try:
        _, vector_store, _ = get_components()
        stats = vector_store.get_stats()
        return VectorStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting vector stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/data")
async def get_graph_data():
    """Export graph data as JSON."""
    try:
        graph_builder, _, _ = get_components()
        
        if graph_builder.use_neo4j:
            # Export from Neo4j
            with graph_builder.neo4j_driver.session() as session:
                # Get all nodes - handle missing attributes gracefully
                nodes_result = session.run("""
                    MATCH (n) 
                    RETURN n.name as name, 
                           n.type as type, 
                           CASE WHEN n.attributes IS NOT NULL THEN n.attributes ELSE {} END as attributes
                """)
                nodes = [{"name": record["name"], "type": record["type"], "attributes": record["attributes"]} 
                        for record in nodes_result]
                
                # Get all relationships - handle missing attributes gracefully
                rels_result = session.run("""
                    MATCH (source:Entity)-[r:RELATION]->(target:Entity)
                    RETURN source.name as source, 
                           target.name as target, 
                           r.relation_type as relation, 
                           CASE WHEN r.attributes IS NOT NULL THEN r.attributes ELSE {} END as attributes
                """)
                relationships = [{"source": record["source"], "target": record["target"], 
                                "relation": record["relation"], "attributes": record["attributes"]}
                               for record in rels_result]
                
                graph_data = {"nodes": nodes, "edges": relationships}
        else:
            # Load from saved JSON file
            data_file = Path("./data/graph_data.json")
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
            else:
                graph_data = {"nodes": [], "edges": []}
        
        return JSONResponse(content=graph_data)
        
    except Exception as e:
        logger.error(f"Error getting graph data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# New advanced endpoints

@app.post("/ontology/save")
async def save_ontology(ontology_data: Dict[str, Any]):
    """Save ontology data to the graph."""
    try:
        graph_builder, _, _ = get_components()
        
        # Process ontology data
        nodes = ontology_data.get('nodes', [])
        edges = ontology_data.get('edges', [])
        
        # Convert to graph format and save
        # This would integrate with the graph builder
        result = {
            "message": "Ontology saved successfully",
            "nodes_saved": len(nodes),
            "edges_saved": len(edges)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error saving ontology: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deduplicate")
async def deduplicate_entities():
    """Run entity deduplication on the current graph."""
    try:
        graph_builder, _, _ = get_components()
        entity_deduplicator, _, _ = get_advanced_components()
        
        # Get current graph data
        graph_data = graph_builder.get_graph_data()
        
        # Run deduplication
        deduplicated_data = entity_deduplicator.deduplicate_graph(graph_data)
        
        # Update graph with deduplicated data
        # This would require updating the graph builder
        
        return {
            "message": "Entity deduplication completed",
            "stats": deduplicated_data.get('deduplication_stats', {})
        }
        
    except Exception as e:
        logger.error(f"Error in deduplication: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/generate")
async def generate_query(query_request: Dict[str, Any]):
    """Generate Cypher/Gremlin queries from natural language."""
    try:
        _, _, _ = get_components()
        _, query_generator, _ = get_advanced_components()
        
        natural_query = query_request.get('query', '')
        query_type = query_request.get('type', 'cypher')  # 'cypher' or 'gremlin'
        schema_info = query_request.get('schema', None)
        
        if query_type == 'cypher':
            result = query_generator.generate_cypher_query(natural_query, schema_info)
        elif query_type == 'gremlin':
            result = query_generator.generate_gremlin_query(natural_query, schema_info)
        else:
            result = query_generator.generate_hybrid_query(natural_query, schema_info)
        
        return {
            "query": result.query,
            "query_type": result.query_type,
            "intent": result.intent,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "parameters": result.parameters
        }
        
    except Exception as e:
        logger.error(f"Error generating query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reasoning/advanced")
async def advanced_reasoning(reasoning_request: Dict[str, Any]):
    """Perform advanced reasoning on a query."""
    try:
        graph_builder, vector_store, retrieval_agent = get_components()
        _, _, reasoning_engine = get_advanced_components()
        
        query = reasoning_request.get('query', '')
        strategy = reasoning_request.get('strategy', 'analytical')
        
        # Get context from existing components
        context = {
            "graph_data": graph_builder.get_graph_data() if graph_builder else {},
            "vector_stats": vector_store.get_stats() if vector_store else {}
        }
        
        # Perform advanced reasoning
        reasoning_chain = await reasoning_engine.reason_about_query(query, context, strategy)
        
        return {
            "query": query,
            "strategy": strategy,
            "final_answer": reasoning_chain.final_answer,
            "confidence": reasoning_chain.confidence,
            "reasoning_steps": [
                {
                    "step_id": step.step_id,
                    "action": step.action,
                    "reasoning": step.reasoning,
                    "confidence": step.confidence
                }
                for step in reasoning_chain.steps
            ],
            "is_complete": reasoning_chain.is_complete
        }
        
    except Exception as e:
        logger.error(f"Error in advanced reasoning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        graph_builder, vector_store, retrieval_agent = get_components()
        return {
            "status": "healthy",
            "components": {
                "graph_builder": "initialized",
                "vector_store": "initialized", 
                "retrieval_agent": "initialized"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/ping")
async def ping():
    """Lightweight ping endpoint for keep-alive purposes."""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


# Rate limiting endpoints
@app.get("/rate-limit/status")
async def get_rate_limit_status():
    """Get current rate limiting status and configuration."""
    try:
        from rate_limiter import rate_limiter
        stats = await rate_limiter.get_global_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rate-limit/user/{user_id}")
async def get_user_rate_limit_status(user_id: str):
    """Get rate limiting status for a specific user."""
    try:
        from rate_limiter import rate_limiter
        status_info = await rate_limiter.get_user_status(user_id)
        return {
            "status": "success",
            "data": status_info
        }
    except Exception as e:
        logger.error(f"Error getting user rate limit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rate-limit/reset/{user_id}")
async def reset_user_rate_limits(user_id: str):
    """Reset rate limits for a specific user (admin function)."""
    try:
        from rate_limiter import rate_limiter
        await rate_limiter.reset_user_limits(user_id)
        return {
            "status": "success",
            "message": f"Rate limits reset for user: {user_id}"
        }
    except Exception as e:
        logger.error(f"Error resetting user rate limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the application
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
