#!/usr/bin/env python3
"""
Configuration validation script for the RAG system

Validates the configuration and environment setup.
TODO: Add more validation checks, maybe test API connectivity
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def validate_environment():
    """Validate environment variables and configuration."""
    print("Validating configuration...")
    
    # Load environment variables
    load_dotenv()
    
    issues = []
    warnings = []
    
    # Check required variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for LLM operations"
    }
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            issues.append(f"ERROR: {var}: {description} (REQUIRED)")
        elif var == "OPENAI_API_KEY" and not value.startswith("sk-"):
            warnings.append(f"WARNING: {var}: Value doesn't look like a valid OpenAI API key")
        else:
            print(f"SUCCESS: {var}: Set")
    
    # Check optional Neo4j variables
    neo4j_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    neo4j_set = all(os.getenv(var) for var in neo4j_vars)
    
    if neo4j_set:
        print("SUCCESS: Neo4j configuration: Complete")
    else:
        print("INFO: Neo4j configuration: Incomplete (will use NetworkX fallback)")
        warnings.append("WARNING: Neo4j not configured - using NetworkX for graph storage")
    
    # Check application variables
    app_vars = {
        "APP_HOST": "0.0.0.0",
        "APP_PORT": "8000",
        "DEBUG": "True",
        "DATA_DIR": "./data",
        "FAISS_INDEX_DIR": "./data/faiss_index"
    }
    
    for var, default in app_vars.items():
        value = os.getenv(var, default)
        print(f"SUCCESS: {var}: {value}")
    
    return issues, warnings

def validate_directories():
    """Validate required directories exist."""
    print("\nValidating directories...")
    
    issues = []
    
    # Check data directory
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    if not data_dir.exists():
        print(f"INFO: Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"SUCCESS: Data directory exists: {data_dir}")
    
    # Check FAISS index directory
    faiss_dir = Path(os.getenv("FAISS_INDEX_DIR", "./data/faiss_index"))
    if not faiss_dir.exists():
        print(f"INFO: Creating FAISS index directory: {faiss_dir}")
        faiss_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"SUCCESS: FAISS index directory exists: {faiss_dir}")
    
    # Check prompts directory
    prompts_dir = Path("prompts")
    if not prompts_dir.exists():
        issues.append("ERROR: Prompts directory not found")
    else:
        print(f"SUCCESS: Prompts directory exists: {prompts_dir}")
        
        # Check prompt files
        prompt_files = ["extract_graph_prompt.txt", "retrieval_decision_prompt.txt"]
        for prompt_file in prompt_files:
            prompt_path = prompts_dir / prompt_file
            if not prompt_path.exists():
                issues.append(f"ERROR: Prompt file not found: {prompt_file}")
            else:
                print(f"SUCCESS: Prompt file exists: {prompt_file}")
    
    return issues

def validate_dependencies():
    """Validate Python dependencies."""
    print("\nValidating dependencies...")
    
    issues = []
    required_packages = [
        "fastapi",
        "uvicorn",
        "langchain",
        "langchain_openai",
        "openai",
        "faiss_cpu",
        "networkx",
        "python_dotenv",
        "pandas",
        "pydantic"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"SUCCESS: {package}: Installed")
        except ImportError:
            issues.append(f"ERROR: {package}: Not installed")
    
    return issues

def validate_sample_data():
    """Validate sample data files."""
    print("\nValidating sample data...")
    
    issues = []
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    
    sample_files = ["python.txt", "ai.txt", "databases.txt"]
    for sample_file in sample_files:
        file_path = data_dir / sample_file
        if not file_path.exists():
            issues.append(f"ERROR: Sample file not found: {sample_file}")
        else:
            print(f"SUCCESS: Sample file exists: {sample_file}")
    
    return issues

def main():
    """Main validation function."""
    print("Agentic Graph RAG Configuration Validator")
    print("=" * 50)
    
    all_issues = []
    all_warnings = []
    
    # Run all validations
    env_issues, env_warnings = validate_environment()
    all_issues.extend(env_issues)
    all_warnings.extend(env_warnings)
    
    dir_issues = validate_directories()
    all_issues.extend(dir_issues)
    
    dep_issues = validate_dependencies()
    all_issues.extend(dep_issues)
    
    data_issues = validate_sample_data()
    all_issues.extend(data_issues)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Validation Summary")
    print("=" * 50)
    
    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"   {warning}")
    
    if all_issues:
        print("\nIssues found:")
        for issue in all_issues:
            print(f"   {issue}")
        
        print(f"\nERROR: Validation failed: {len(all_issues)} issues found")
        print("\nTo fix these issues:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Configure your .env file with required variables")
        print("   3. Ensure all required files are present")
        return 1
    else:
        print("\nSUCCESS: Configuration validation passed!")
        print("\nYou can now start the service with: python run.py")
        return 0

if __name__ == "__main__":
    sys.exit(main())
