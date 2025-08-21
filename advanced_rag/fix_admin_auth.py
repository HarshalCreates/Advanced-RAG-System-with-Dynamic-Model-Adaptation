#!/usr/bin/env python3
"""
Quick fix for admin authentication issues.
This script helps diagnose and fix admin API authentication problems.
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.config import get_settings

def check_admin_config():
    """Check current admin configuration."""
    print("🔧 Checking Admin Configuration...")
    
    settings = get_settings()
    
    print(f"📋 Configuration Status:")
    print(f"  • Admin API Key: {'✅ SET' if settings.admin_api_key else '❌ NOT SET'}")
    if settings.admin_api_key:
        print(f"  • Key Value: {'*' * len(settings.admin_api_key)}")
    
    print(f"  • Environment Variables:")
    rag_admin_key = os.getenv("RAG_ADMIN_API_KEY")
    print(f"    - RAG_ADMIN_API_KEY: {'✅ SET' if rag_admin_key else '❌ NOT SET'}")
    
    admin_api_key = os.getenv("ADMIN_API_KEY")
    print(f"    - ADMIN_API_KEY: {'✅ SET' if admin_api_key else '❌ NOT SET'}")
    
    return settings

def fix_admin_auth():
    """Provide solutions for admin auth issues."""
    print("\n🛠️ Admin Authentication Solutions:")
    
    print("\n**Option 1: Disable Authentication (Development Mode)**")
    print("Set an empty admin API key in your .env file:")
    print("```")
    print("ADMIN_API_KEY=")
    print("RAG_ADMIN_API_KEY=")
    print("```")
    
    print("\n**Option 2: Set Admin API Key**")
    print("Set a secure admin API key:")
    print("```")
    print("ADMIN_API_KEY=your_secure_admin_key_here")
    print("RAG_ADMIN_API_KEY=your_secure_admin_key_here")
    print("```")
    
    print("\n**Option 3: Use Environment Variables**")
    print("Export the variables in your shell:")
    print("```bash")
    print("export RAG_ADMIN_API_KEY=your_key_here")
    print("export ADMIN_API_KEY=your_key_here")
    print("```")
    
    print("\n**Option 4: Quick Fix (Temporary)**")
    print("For immediate testing, create a .env file:")
    
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        try:
            with env_file.open("w") as f:
                f.write("# RAG System Configuration\n")
                f.write("ADMIN_API_KEY=\n")
                f.write("RAG_ADMIN_API_KEY=\n")
                f.write("# Leave empty for development mode (no auth required)\n")
            
            print(f"✅ Created .env file at: {env_file}")
            print("   Admin authentication is now disabled for development.")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
    else:
        print(f"📄 .env file already exists at: {env_file}")

def test_api_call():
    """Test the admin API call."""
    print("\n🧪 Testing Admin API Call...")
    
    try:
        import httpx
        
        # Test without API key first
        with httpx.Client() as client:
            response = client.post(
                "http://127.0.0.1:8000/api/admin/hot-swap/generation",
                params={"backend": "ollama", "model": "llama3.2:3b"},
                timeout=10
            )
            
            print(f"📡 API Response:")
            print(f"  • Status Code: {response.status_code}")
            print(f"  • Response: {response.text}")
            
            if response.status_code == 200:
                print("✅ Model hot-swap successful!")
            elif response.status_code == 401:
                print("❌ Authentication required - use one of the fixes above")
            else:
                print(f"⚠️ Unexpected response: {response.status_code}")
                
    except Exception as e:
        print(f"❌ API test failed: {e}")
        print("💡 Make sure the RAG server is running on port 8000")

if __name__ == "__main__":
    print("🚀 RAG Admin Authentication Fixer")
    print("=" * 50)
    
    # Check current config
    settings = check_admin_config()
    
    # Provide solutions
    fix_admin_auth()
    
    # Test the API
    test_api_call()
    
    print("\n🎉 After applying the fix, try switching models in Chainlit again!")
