#!/usr/bin/env python3
"""
Simple script to set your OpenAI API key in the .env file.
"""

import os
from pathlib import Path

def main():
    print("🔑 OpenAI API Key Configuration")
    print("=" * 35)
    
    # Get API key from user
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided.")
        return
    
    # Create .env file content
    env_content = f"""# OpenAI API Key (REQUIRED for ChatGPT/GPT-4)
OPENAI_API_KEY={api_key}

# Embedding Configuration (already fixed)
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Generation Configuration
GENERATION_BACKEND=openai
GENERATION_MODEL=gpt-4o

# Other settings
RETRIEVER_BACKEND=faiss
LOG_LEVEL=INFO
ENVIRONMENT=development
"""
    
    # Write to .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"✅ API key saved to .env file!")
    print("\n📋 Next steps:")
    print("1. Restart the server:")
    print("   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
    print("\n2. Test with CLI:")
    print('   python cli_query.py -q "What is machine learning?" --citations')
    print("\n🎯 Your system will now use ChatGPT for generating answers!")

if __name__ == "__main__":
    main()
