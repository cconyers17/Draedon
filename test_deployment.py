#!/usr/bin/env python3
"""
Test script to verify deployment readiness for Text-to-CAD application.
Checks both frontend and backend configurations.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def test_frontend():
    """Test frontend build and configuration."""
    print("Testing Frontend...")

    frontend_dir = Path("text-to-cad-app")

    # Check if package.json exists
    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print("FAIL: package.json not found")
        return False

    # Check build artifacts
    next_dir = frontend_dir / ".next"
    if not next_dir.exists():
        print("WARNING: .next directory not found, run 'npm run build' first")
        return False

    # Check environment files
    env_prod = frontend_dir / ".env.production"
    if env_prod.exists():
        print("PASS: Production environment file found")
    else:
        print("WARNING: .env.production not found")

    print("PASS: Frontend checks passed")
    return True

def test_backend():
    """Test backend configuration and dependencies."""
    print("🔧 Testing Backend...")

    backend_dir = Path("backend")

    # Check requirements.txt
    requirements = backend_dir / "requirements.txt"
    if not requirements.exists():
        print("❌ requirements.txt not found")
        return False

    # Check main app file
    main_py = backend_dir / "app" / "main.py"
    if not main_py.exists():
        print("❌ app/main.py not found")
        return False

    # Check critical services
    cad_service = backend_dir / "app" / "services" / "cad" / "cad_service.py"
    nlp_service = backend_dir / "app" / "services" / "nlp" / "nlp_service.py"

    if cad_service.exists():
        print("✅ CAD service found")
    else:
        print("❌ CAD service missing")
        return False

    if nlp_service.exists():
        print("✅ NLP service found")
    else:
        print("❌ NLP service missing")
        return False

    # Check environment files
    env_prod = backend_dir / ".env.production"
    if env_prod.exists():
        print("✅ Production environment file found")
    else:
        print("⚠️  .env.production not found")

    print("✅ Backend checks passed")
    return True

def test_deployment_configs():
    """Test deployment configuration files."""
    print("🔧 Testing Deployment Configs...")

    # Check Render configs
    render_frontend = Path("text-to-cad-app") / "render-frontend.yaml"
    render_backend = Path("backend") / "render-backend.yaml"

    if render_frontend.exists():
        print("✅ Frontend Render config found")
    else:
        print("⚠️  Frontend Render config missing")

    if render_backend.exists():
        print("✅ Backend Render config found")
    else:
        print("⚠️  Backend Render config missing")

    # Check deployment guide
    deploy_guide = Path("DEPLOYMENT_GUIDE.md")
    if deploy_guide.exists():
        print("✅ Deployment guide found")
    else:
        print("⚠️  Deployment guide missing")

    print("✅ Deployment config checks passed")
    return True

def main():
    """Run all deployment tests."""
    print("Text-to-CAD Deployment Readiness Test")
    print("=" * 50)

    tests = [
        ("Frontend", test_frontend),
        ("Backend", test_backend),
        ("Deployment Configs", test_deployment_configs)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            print()
        except Exception as e:
            print(f"❌ {name} test failed: {e}")
            results.append((name, False))
            print()

    print("📊 Test Results:")
    print("=" * 30)
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("\n🎯 Deployment Readiness:")
    if all_passed:
        print("✅ Ready for deployment!")
        print("\n📋 Next Steps:")
        print("1. Push code to GitHub")
        print("2. Create Render services using the provided configs")
        print("3. Set environment variables in Render dashboard")
        print("4. Deploy and test")
    else:
        print("❌ Issues found - please fix before deploying")
        print("\n🔧 Common fixes:")
        print("- Run 'npm run build' in text-to-cad-app directory")
        print("- Ensure all required files are present")
        print("- Check configuration files")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)