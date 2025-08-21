#!/usr/bin/env python3
"""
Simple script to set up your OpenAI API key for the RAG system.
"""

import os
from pathlib import Path

def main():
    print("🔑 OpenAI API Key Setup for Advanced RAG")
    print("=" * 45)
    print()
    print("This will configure your system to use OpenAI for generation instead of the fallback mode.")
    print()
    
    # Check if .env already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists. This will overwrite it.")
        response = input("Continue? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Get API key from user
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. System will continue using fallback generation.")
        return
    
    # Create .env file content
    env_content = f"""# OpenAI API Key (REQUIRED for ChatGPT/GPT-4 generation)
OPENAI_API_KEY={api_key}

# Embedding Configuration (using local models for privacy)
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Generation Configuration (now using OpenAI)
GENERATION_BACKEND=openai
GENERATION_MODEL=gpt-4o

# Retrieval Configuration
RETRIEVER_BACKEND=faiss
INDEX_DIR=./data/index
UPLOADS_DIR=./data/uploads

# Other settings
LOG_LEVEL=INFO
ENVIRONMENT=development
ALLOWED_ORIGINS=*
"""
    
    # Write to .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"✅ API key saved to .env file!")
    print()
    print("📋 Next steps:")
    print("1. Restart the server:")
    print("   python start_server.py")
    print("   # or")
    print("   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
    print()
    print("2. Test with Chainlit UI:")
    print("   chainlit run ui/chainlit/langchain_app.py")
    print()
    print("3. Test with CLI:")
    print('   python cli_query.py -q "What is machine learning?" --citations')
    print()
    print("🎯 Your system will now use ChatGPT for generating answers!")
    print("💡 If you don't have an API key, the system will continue working with fallback generation.")

if __name__ == "__main__":
    main()
