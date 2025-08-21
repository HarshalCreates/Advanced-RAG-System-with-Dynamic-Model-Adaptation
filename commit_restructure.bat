@echo off
echo Committing restructured repository...

echo Adding all changes...
git add .

echo Removing old advanced_rag references...
git rm -r advanced_rag --ignore-unmatch

echo Creating commit...
git commit -m "Restructure repository: Move all files to root level

- Moved all project files from advanced_rag/ folder to root directory
- All main components now visible on GitHub main page:
  * app/ - Main FastAPI application
  * data/ - Data and uploads
  * docs/ - Documentation
  * ui/ - Chainlit interface
  * docker/ - Docker configurations
- Removed nested folder structure for better GitHub presentation
- All features now immediately accessible at repository root"

echo Pushing to GitHub...
git push origin main

echo Repository restructure completed!
pause
