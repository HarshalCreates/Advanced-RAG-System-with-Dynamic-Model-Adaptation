#!/usr/bin/env python3
"""Create production .env file."""

env_content = """# Production RAG Configuration
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
GENERATION_BACKEND=echo
GENERATION_MODEL=gpt-4o
RETRIEVER_BACKEND=faiss
LOG_LEVEL=INFO
ENVIRONMENT=production
TF_ENABLE_ONEDNN_OPTS=0
TF_CPP_MIN_LOG_LEVEL=2
"""

with open('.env', 'w') as f:
    f.write(env_content)

print("✅ Production .env file created")
