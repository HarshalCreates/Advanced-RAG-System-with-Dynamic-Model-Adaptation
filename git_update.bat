@echo off
echo Starting Git operations...

echo Adding files to git...
git add .gitignore
git add README.md
git add start_rag_server.py
git add git_update.bat
git add advanced_rag/
git add data/

echo Current git status:
git status

echo Creating commit...
git commit -m "Advanced RAG System - Initial commit

Features included:
- Multi-modal document processing (PDF, images, text)
- Advanced retrieval strategies (hybrid, graph-based, semantic)
- Multiple embedding models support (OpenAI, Cohere, HuggingFace)
- Docker containerization
- Chainlit UI interface
- Comprehensive evaluation and monitoring
- Security and privacy features
- API endpoints for all functionality"

echo Commit completed!
echo Current git log:
git log --oneline -5

pause
