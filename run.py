#!/usr/bin/env python3
"""
Startup script for the RAG system

Easy way to start the service with proper configuration.
TODO: Add better error handling, maybe a config validator
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



def check_requirements():
    """Check if all requirements are met."""
    print("Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ is required")
        return False
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("ERROR: .env file not found")
        print("   Please copy env_template.txt to .env and configure it")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return False
    
    print("SUCCESS: All requirements met")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("SUCCESS: Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return False

def start_service():
    """Start the FastAPI service."""
    print("Starting Agentic Graph RAG Service...")
    
    # Get configuration
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"   API Documentation: http://{host}:{port}/docs")
    print(f"   Health Check: http://{host}:{port}/health")
    print("\n" + "="*50)
    
    try:
        # Start the service
        import uvicorn
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nService stopped by user")
    except Exception as e:
        print(f"ERROR: Error starting service: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("Agentic Graph RAG as a Service")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        print("\nERROR: Requirements check failed. Please fix the issues above.")
        return 1
    
    # Skip dependency installation in non-interactive mode
    # install_deps = input("\nInstall/update dependencies? (y/N): ").lower().strip()
    # if install_deps in ['y', 'yes']:
    #     if not install_dependencies():
    #         print("\nERROR: Failed to install dependencies")
    #         return 1
    
    # Start the service
    print("\nStarting service...")
    if not start_service():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
