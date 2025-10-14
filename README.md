# Agentic Graph RAG as a Service

A RAG platform that combines knowledge graphs with vector search for document retrieval and question answering.

*Note: This is a working prototype. For production use, consider adding authentication, rate limiting, and better error handling.*

## Features

- **Document Processing**: Converts unstructured documents into knowledge graphs
- **Dual Retrieval**: Combines FAISS vector search with graph-based retrieval
- **Intelligent Agent**: Dynamically decides between graph, vector, or hybrid retrieval
- **REST API**: FastAPI service with streaming support
- **Multiple Databases**: Supports both Neo4j and NetworkX for graph storage
- **Production Ready**: Modular, well-documented, and scalable architecture

### Advanced Features
- **Frontend UI**: Visual ontology editor with interactive graph visualization
- **Entity Deduplication**: Advanced algorithms for identifying and merging duplicate entities
- **Query Generation**: Automatic Cypher and Gremlin query generation from natural language
- **Advanced Reasoning**: Multi-step reasoning chains with iterative refinement
- **Ontology Management**: Save, load, and edit knowledge graph ontologies

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Graph Builder   │───▶│   Knowledge     │
│   (TXT/PDF/MD)  │    │  (Triple Extract)│    │     Graph       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Vector Store    │
                       │  (FAISS + OpenAI)│
                       └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ Retrieval Agent  │───▶│   Response      │
│                 │    │ (Decision Logic) │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+ (tested with 3.10, might work with 3.9)
- OpenAI API key (get one from openai.com)
- Neo4j (optional, falls back to NetworkX if not available)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd agentic_graph_rag
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp env_template.txt .env
   # Edit .env with your API keys and configuration
   ```

4. **Configure your .env file:**
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   NEO4J_URI=bolt://localhost:7687  # Optional
   NEO4J_USERNAME=neo4j             # Optional
   NEO4J_PASSWORD=your_password     # Optional
   APP_HOST=0.0.0.0
   APP_PORT=8000
   DEBUG=True
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## Usage

### 1. Ingest Documents

Upload documents to the knowledge base:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@python.txt" \
  -F "files=@ai.txt" \
  -F "files=@databases.txt" \
  -F "process_existing=true"
```

### 2. Query the Knowledge Base

Ask questions using natural language:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Who created Python?", "stream": false}'
```

### 3. Stream Responses

Get real-time streaming responses:

```bash
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the connection between AI and Python?"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/query` | POST | Query the knowledge base |
| `/query/stream` | POST | Stream query responses |
| `/ingest` | POST | Upload and process documents |
| `/graph/stats` | GET | Knowledge graph statistics |
| `/vector/stats` | GET | Vector store statistics |
| `/graph/data` | GET | Export graph data as JSON |
| `/health` | GET | Health check |

## Retrieval Strategies

The system intelligently chooses retrieval strategies based on query analysis:

### Graph Retrieval
- **Triggers**: Relationship queries, entity connections
- **Examples**: "How are X and Y connected?", "Who worked with whom?"
- **Benefits**: Exploits entity relationships and graph structure

### Vector Retrieval
- **Triggers**: Semantic meaning, explanations, definitions
- **Examples**: "What does X mean?", "Explain the concept of Y"
- **Benefits**: Captures semantic similarity and context

### Hybrid Retrieval
- **Triggers**: Complex questions requiring both approaches
- **Examples**: "How does AI relate to Python programming?"
- **Benefits**: Combines relational and semantic understanding

## Project Structure

```
agentic_graph_rag/
├── app.py                  # FastAPI service
├── agent.py                # Retrieval agent logic
├── graph_builder.py        # Document-to-graph pipeline
├── vector_store.py         # FAISS vector operations
├── prompts/
│   ├── extract_graph_prompt.txt
│   └── retrieval_decision_prompt.txt
├── data/
│   ├── faiss_index/        # FAISS index storage
│   ├── graph_data.json     # Graph data export
│   ├── python.txt          # Sample data
│   ├── ai.txt              # Sample data
│   └── databases.txt       # Sample data
├── requirements.txt
├── env_template.txt
└── README.md
```

## Example Queries

Test the system with these sample queries:

1. **Graph Retrieval**: "Who created Python?"
2. **Vector Retrieval**: "Explain the main differences between SQL and NoSQL"
3. **Hybrid Retrieval**: "What's the connection between AI and Python?"

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `NEO4J_URI` | Neo4j database URI | Optional |
| `NEO4J_USERNAME` | Neo4j username | Optional |
| `NEO4J_PASSWORD` | Neo4j password | Optional |
| `APP_HOST` | Application host | 0.0.0.0 |
| `APP_PORT` | Application port | 8000 |
| `DEBUG` | Debug mode | True |
| `DATA_DIR` | Data directory | ./data |
| `FAISS_INDEX_DIR` | FAISS index directory | ./data/faiss_index |

### Graph Database Options

1. **Neo4j** (Recommended for production):
   - Install Neo4j Desktop or use Neo4j AuraDB
   - Configure connection in .env file
   - Supports complex graph queries and relationships

2. **NetworkX** (Default fallback):
   - No additional setup required
   - Suitable for development and small datasets
   - Data exported to JSON format

## Testing

Run the system with sample data:

1. **Start the service:**
   ```bash
   python app.py
   ```

2. **Ingest sample documents:**
   ```bash
   curl -X POST "http://localhost:8000/ingest" \
     -F "process_existing=true"
   ```

3. **Test queries:**
   ```bash
   # Graph retrieval
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Who created Python?"}'
   
   # Vector retrieval
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain the main differences between SQL and NoSQL"}'
   
   # Hybrid retrieval
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the connection between AI and Python?"}'
   ```

## Advanced Features

### Frontend Ontology Editor

Access the visual ontology editor at: http://localhost:8000/editor

Features:
- Interactive graph visualization
- Add/edit entities and relationships
- Real-time graph updates
- Export/import ontology data

### Entity Deduplication

Run entity deduplication to merge duplicate entities:

```bash
curl -X POST "http://localhost:8000/deduplicate"
```

### Query Generation

Generate Cypher or Gremlin queries from natural language:

```bash
# Generate Cypher query
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "Find all people who work at companies", "type": "cypher"}'

# Generate Gremlin query
curl -X POST "http://localhost:8000/query/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show relationships between technologies", "type": "gremlin"}'
```

### Advanced Reasoning

Perform multi-step reasoning with different strategies:

```bash
# Analytical reasoning
curl -X POST "http://localhost:8000/reasoning/advanced" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main themes in AI research?", "strategy": "analytical"}'

# Creative reasoning
curl -X POST "http://localhost:8000/reasoning/advanced" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do different programming languages relate?", "strategy": "creative"}'
```

### Ontology Management

Save and manage ontology data:

```bash
# Save ontology
curl -X POST "http://localhost:8000/ontology/save" \
  -H "Content-Type: application/json" \
  -d '{"nodes": [...], "edges": [...]}'
```

### Testing Advanced Features

Run the comprehensive test suite:

```bash
python test_advanced_features.py
```

## Rate Limiting & API Protection

The system includes comprehensive rate limiting to protect against API abuse, cost overruns, and ensure fair usage. The rate limiter implements multiple strategies for robust protection.

### 🛡️ Rate Limiting Features

- **Multi-tier Protection**: Per-minute, per-hour, and per-day limits
- **Token-based Limits**: Tracks OpenAI API token usage and costs
- **Burst Protection**: Prevents rapid-fire requests
- **Cost Monitoring**: Tracks and limits daily API costs
- **User Identification**: IP-based and API key-based user tracking
- **Exempt Paths**: Health checks and documentation endpoints are exempt
- **Real-time Headers**: Rate limit status in HTTP response headers

### 📊 Rate Limit Configuration

Default limits (configurable in `rate_limiter.py`):

| Limit Type | Default Value | Description |
|------------|---------------|-------------|
| **Per Minute** | 10 requests, 50K tokens | Prevents burst abuse |
| **Per Hour** | 100 requests, 200K tokens | Hourly usage control |
| **Per Day** | 1000 requests, 1M tokens | Daily usage limits |
| **Daily Cost** | $10.00 USD | Cost protection |
| **Burst Window** | 5 requests in 10 seconds | Immediate burst protection |

### 🔧 Rate Limiter Components

#### 1. **RateLimiter Class** (`rate_limiter.py`)
Core rate limiting logic with multiple protection layers:

```python
# Check rate limits before processing
allowed, reason, status_info = await rate_limiter.check_rate_limit(
    user_id="ip:192.168.1.1",
    estimated_tokens=1000,
    model='gpt-3.5-turbo'
)
```

#### 2. **RateLimitMiddleware** (`rate_limit_middleware.py`)
ASGI middleware for automatic rate limiting:

```python
# Applied to all endpoints except exempt paths
app.add_middleware(
    RateLimitMiddleware,
    exempt_paths=['/health', '/docs', '/openapi.json']
)
```

#### 3. **User Identification Methods**
Multiple ways to identify users:

- **IP-based**: `ip:192.168.1.1`
- **API Key**: `api_key:your_api_key`
- **Auth Token**: `auth:bearer_token`

### 📈 Rate Limit Monitoring

#### Check Global Statistics
```bash
curl -X GET "http://localhost:8000/rate-limit/status"
```

Response:
```json
{
  "status": "success",
  "data": {
    "total_users": 15,
    "active_users": 3,
    "total_requests_today": 245,
    "total_tokens_today": 125000,
    "total_cost_today": 1.25,
    "config": {
      "requests_per_minute": 10,
      "requests_per_hour": 100,
      "requests_per_day": 1000,
      "tokens_per_minute": 50000,
      "tokens_per_hour": 200000,
      "tokens_per_day": 1000000,
      "cost_per_day": 10.0
    }
  }
}
```

#### Check User-Specific Status
```bash
curl -X GET "http://localhost:8000/rate-limit/user/ip:192.168.1.1"
```

Response:
```json
{
  "status": "success",
  "data": {
    "requests_this_minute": 3,
    "requests_this_hour": 15,
    "requests_this_day": 45,
    "tokens_this_minute": 5000,
    "tokens_this_hour": 25000,
    "tokens_this_day": 75000,
    "cost_this_day": 0.75,
    "last_request": 1703123456.789,
    "limits": {
      "requests_per_minute": 10,
      "requests_per_hour": 100,
      "requests_per_day": 1000,
      "tokens_per_minute": 50000,
      "tokens_per_hour": 200000,
      "tokens_per_day": 1000000,
      "cost_per_day": 10.0
    }
  }
}
```

### 🚫 Rate Limit Responses

When limits are exceeded, the API returns:

**HTTP Status**: `429 Too Many Requests`

**Response Body**:
```json
{
  "error": "Rate limit exceeded",
  "message": "Rate limit exceeded: too many requests or tokens per minute.",
  "retry_after": 60,
  "status_info": {
    "requests_this_minute": 10,
    "requests_this_hour": 45,
    "tokens_this_minute": 50000,
    "cost_this_day": 2.50
  }
}
```

**Response Headers**:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1703123600
X-RateLimit-Status: exceeded
```

### ⚙️ Configuration & Customization

#### 1. **Modify Rate Limits**
Edit `rate_limiter.py`:

```python
@dataclass
class RateLimitConfig:
    # Adjust these values for your needs
    requests_per_minute: int = 20      # Increase for higher traffic
    tokens_per_minute: int = 100000    # Increase for larger requests
    requests_per_hour: int = 200       # Hourly limit
    tokens_per_hour: int = 500000      # Hourly token limit
    requests_per_day: int = 2000       # Daily limit
    tokens_per_day: int = 2000000      # Daily token limit
    cost_per_day: float = 20.0         # Daily cost limit in USD
    max_burst_requests: int = 10       # Burst protection
    burst_window_seconds: int = 15     # Burst window
```

#### 2. **Custom User Identification**
Create custom user ID extraction:

```python
def get_user_id_from_custom_header(request: Request) -> str:
    """Extract user ID from custom header."""
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return f"user:{user_id}"
    
    # Fallback to IP
    return f"ip:{request.client.host}"

# Apply custom identification
app.add_middleware(
    RateLimitMiddleware,
    get_user_id=get_user_id_from_custom_header,
    exempt_paths=['/health', '/docs', '/openapi.json']
)
```

#### 3. **Exempt Additional Paths**
Add more exempt paths:

```python
app.add_middleware(
    RateLimitMiddleware,
    exempt_paths=[
        '/health', 
        '/docs', 
        '/openapi.json',
        '/graph/stats',
        '/vector/stats',
        '/admin/metrics'  # Add custom exempt paths
    ]
)
```

### 🔄 Admin Functions

#### Reset User Rate Limits
```bash
# Reset limits for a specific user (admin function)
curl -X POST "http://localhost:8000/rate-limit/reset/ip:192.168.1.1"
```

#### Monitor Rate Limit Health
```bash
# Check if rate limiting is working
curl -X GET "http://localhost:8000/health"
```

### 🎯 Best Practices

#### 1. **Production Configuration**
For production environments:

```python
# More restrictive limits for production
RateLimitConfig(
    requests_per_minute=5,      # Conservative
    tokens_per_minute=25000,    # Lower token limit
    requests_per_hour=50,       # Hourly limit
    tokens_per_hour=100000,     # Hourly tokens
    requests_per_day=500,      # Daily limit
    tokens_per_day=500000,     # Daily tokens
    cost_per_day=5.0,          # Lower cost limit
    max_burst_requests=3,      # Strict burst protection
    burst_window_seconds=10    # Shorter burst window
)
```

#### 2. **Monitoring & Alerting**
Set up monitoring for rate limit violations:

```python
# Add logging for rate limit violations
import logging
logger = logging.getLogger(__name__)

# In your application
if not allowed:
    logger.warning(f"Rate limit exceeded for {user_id}: {reason}")
    # Send alert to monitoring system
    send_alert(f"Rate limit exceeded: {reason}")
```

#### 3. **Graceful Degradation**
Handle rate limits gracefully in client applications:

```python
import requests
import time

def make_request_with_retry(url, data, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data)
            
            if response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get('X-RateLimit-Reset', 60))
                time.sleep(retry_after)
                continue
                
            return response.json()
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 🚨 Troubleshooting Rate Limits

#### Common Issues:

1. **"Rate limit exceeded" errors**:
   - Check current usage: `GET /rate-limit/status`
   - Wait for limits to reset
   - Consider increasing limits if legitimate usage

2. **High token usage**:
   - Optimize prompts to reduce token count
   - Use smaller models for simple tasks
   - Implement request batching

3. **Cost overruns**:
   - Monitor daily costs via API
   - Set lower cost limits
   - Use cheaper models when possible

#### Debug Rate Limiting:
```bash
# Check if rate limiting is active
curl -X GET "http://localhost:8000/rate-limit/status"

# Check specific user
curl -X GET "http://localhost:8000/rate-limit/user/ip:127.0.0.1"

# Test with a request to see headers
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' \
  -v  # Shows response headers including rate limit info
```

### 📊 Rate Limit Analytics

The system provides comprehensive analytics:

- **User Activity**: Track individual user usage patterns
- **Cost Analysis**: Monitor API costs and spending trends
- **Peak Usage**: Identify high-traffic periods
- **Violation Tracking**: Monitor rate limit violations
- **Performance Impact**: Measure rate limiting overhead

Access analytics via the monitoring endpoints or integrate with external monitoring systems.

## Production Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "app.py"]
```

### Environment Setup

1. **Production .env:**
   ```env
   OPENAI_API_KEY=your_production_key
   NEO4J_URI=your_neo4j_uri
   NEO4J_USERNAME=your_username
   NEO4J_PASSWORD=your_password
   APP_HOST=0.0.0.0
   APP_PORT=8000
   DEBUG=False
   ```

2. **Run with Gunicorn:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
   ```

## Customization

### Adding New Document Types

Extend the `GraphBuilder` class to support additional file formats:

```python
def _process_single_document(self, file_path: Path):
    if file_path.suffix.lower() == '.pdf':
        # Add PDF processing logic
        pass
    # ... existing code
```

### Custom Retrieval Logic

Modify the `RetrievalAgent` class to implement custom retrieval strategies:

```python
def decide_retrieval_mode(self, query: str) -> str:
    # Add your custom logic here
    pass
```

### Custom Prompts

Update the prompt files in the `prompts/` directory to customize:
- Entity and relationship extraction
- Retrieval decision logic
- Answer generation

## Monitoring

Monitor the system using the built-in endpoints:

- **Health Check**: `GET /health`
- **Graph Stats**: `GET /graph/stats`
- **Vector Stats**: `GET /vector/stats`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Troubleshooting

This section covers common issues encountered during development and their solutions.

### 🚨 **Critical Issue: Neo4j Data Type Errors**

**Problem:** During document ingestion, you may encounter errors like:
```
ERROR:graph_builder:Error storing entity [EntityName]: {neo4j_code: Neo.ClientError.Statement.TypeError} {message: Property values can only be of primitive types or arrays thereof. Encountered: Map{}.}
```

**Root Cause:** Neo4j can only store primitive types (strings, numbers, booleans) or arrays of primitives. Complex dictionary objects cannot be stored directly as properties.

**Solution Applied:**
1. **Flattened Attributes**: Instead of storing complex objects as single properties, we now store each attribute as a separate Neo4j property
2. **Key Sanitization**: Added sanitization to handle spaces and special characters in property names
3. **JSON Serialization**: Complex objects are converted to JSON strings before storage

**Code Fix in `graph_builder.py`:**
```python
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
```

### 🔧 **Service Startup Issues**

**Problem:** Service fails to start or shows "Not Found" errors for endpoints.

**Common Causes & Solutions:**

1. **Import Errors:**
   ```
   ERROR: Error loading ASGI app. Could not import module "app".
   ```
   **Solution:** Ensure you're running from the correct directory:
   ```bash
   cd agentic_graph_rag
   python app.py
   ```

2. **Missing Dependencies:**
   ```
   ModuleNotFoundError: No module named 'fastapi'
   ```
   **Solution:** Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. **Port Already in Use:**
   ```
   ERROR: [Errno 10048] Only one usage of each socket address
   ```
   **Solution:** Kill existing processes or change port:
   ```bash
   # Windows
   taskkill /F /IM python.exe
   
   # Or change port in .env
   APP_PORT=8001
   ```

### 🌐 **Connection Issues**

**Problem:** Cannot connect to service endpoints.

**Solutions:**

1. **Use 127.0.0.1 instead of localhost:**
   ```bash
   # Instead of: http://localhost:8000
   # Use: http://127.0.0.1:8000
   ```

2. **Check Windows Firewall:**
   - Allow Python through Windows Defender Firewall
   - Add exception for port 8000

3. **PowerShell Syntax Issues:**
   ```powershell
   # Wrong (PowerShell doesn't support &&)
   cd agentic_graph_rag && python app.py
   
   # Correct
   cd agentic_graph_rag
   python app.py
   ```

### 📁 **Static Files Issues**

**Problem:** `/editor` endpoint returns "Not Found" error.

**Solution Checklist:**
1. **Verify static directory exists:**
   ```bash
   ls agentic_graph_rag/static/
   # Should show: index.html
   ```

2. **Check FastAPI static mounting:**
   ```python
   # In app.py
   app.mount("/static", StaticFiles(directory="static"), name="static")
   ```

3. **Verify endpoint definition:**
   ```python
   @app.get("/editor")
   async def ontology_editor():
       return FileResponse("static/index.html")
   ```

### 🔄 **Service Restart Issues**

**Problem:** Code changes not being picked up by running service.

**Solutions:**

1. **Force Restart:**
   ```bash
   # Kill all Python processes
   taskkill /F /IM python.exe
   
   # Start fresh
   python run.py
   ```

2. **Use Auto-reload:**
   ```bash
   # The service should auto-reload, but if not:
   python -c "import uvicorn; uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)"
   ```

### 🧪 **Testing Issues**

**Problem:** Query endpoints timeout or fail.

**Solutions:**

1. **Use Web Interface (Recommended):**
   - Go to: http://127.0.0.1:8000/docs
   - Use the interactive API interface

2. **Check Service Health:**
   ```bash
   # Test health first
   Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -Method GET
   ```

3. **PowerShell Query Testing:**
   ```powershell
   $body = @{
       query = "What is Python programming language?"
       stream = $false
   } | ConvertTo-Json

   Invoke-WebRequest -Uri "http://127.0.0.1:8000/query" -Method POST -Body $body -ContentType "application/json"
   ```

### 🔍 **Debugging Steps**

**When things go wrong:**

1. **Check Service Status:**
   ```bash
   # Check if service is running
   Get-Process python -ErrorAction SilentlyContinue
   
   # Check port usage
   netstat -an | findstr :8000
   ```

2. **Review Logs:**
   - Look for ERROR messages in terminal output
   - Check for import errors or connection failures
   - Verify environment variables are loaded

3. **Test Components Individually:**
   ```bash
   # Test health endpoint
   curl http://127.0.0.1:8000/health
   
   # Test graph stats
   curl http://127.0.0.1:8000/graph/stats
   
   # Test vector stats
   curl http://127.0.0.1:8000/vector/stats
   ```

4. **Environment Verification:**
   ```bash
   # Check if .env file exists and has correct values
   cat .env
   
   # Test OpenAI API key
   python -c "import openai; print('API key loaded')"
   ```

### 🚀 **Performance Issues**

**Problem:** Slow response times or memory issues.

**Solutions:**

1. **OpenMP Warning (Safe to Ignore):**
   ```
   OMP: Error #15: Initializing libomp140.x86_64.dll, but found libiomp5md.dll already initialized.
   ```
   **Solution 1 (Environment Variable):**
   ```bash
   set KMP_DUPLICATE_LIB_OK=TRUE
   ```
   
   **Solution 2 (Code Fix - Recommended):**
   Add this at the top of your `run.py` file:
   ```python
   import os
   os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
   
   # Rest of your imports and code...
   ```

2. **Memory Optimization:**
   - Use smaller chunk sizes for large documents
   - Implement document preprocessing
   - Consider batch processing for large datasets

### 📋 **Common Error Messages & Solutions**

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `"StaticFiles" is not defined` | Missing import | Add `from fastapi.staticfiles import StaticFiles` |
| `{"detail":"Not Found"}` | Missing endpoint/static files | Check endpoint definition and file paths |
| `Property values can only be of primitive types` | Neo4j data type error | Use the flattened attributes approach |
| `Unable to connect to the remote server` | Service not running | Start service and check port |
| `can't open file 'app.py'` | Wrong directory | Navigate to correct directory first |

### 🎯 **Quick Fixes Summary**

1. **Service won't start:** Check directory, install dependencies, kill existing processes
2. **Neo4j errors:** Use the flattened attributes approach in graph_builder.py
3. **Connection issues:** Use 127.0.0.1 instead of localhost, check firewall
4. **Static files not found:** Verify static directory and FastAPI mounting
5. **Code changes not applied:** Force restart the service
6. **Query timeouts:** Use web interface at /docs for testing

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the logs for error messages
3. Ensure all environment variables are set correctly
4. Verify OpenAI API key is valid and has sufficient credits
5. Follow the troubleshooting steps above

## Future Enhancements

- [ ] Advanced graph visualization
- [ ] Multi-language support
- [ ] Caching layer for improved performance
- [ ] Authentication and authorization
- [ ] Batch processing capabilities
- [ ] Advanced analytics and insights


## License

Copyright (c) 2025 Nigash M

All Rights Reserved.

Unauthorized copying, modification, distribution, or use of this software,
via any medium, is strictly prohibited without the express permission of the author.
