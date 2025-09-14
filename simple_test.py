#!/usr/bin/env python3
"""Simple deployment readiness test."""

import os
from pathlib import Path

def main():
    print("Text-to-CAD Deployment Readiness Test")
    print("=" * 40)

    # Test Frontend
    print("\n1. Frontend Tests:")
    frontend_dir = Path("text-to-cad-app")

    if (frontend_dir / "package.json").exists():
        print("   PASS: package.json found")
    else:
        print("   FAIL: package.json missing")
        return False

    if (frontend_dir / ".next").exists():
        print("   PASS: Build artifacts found")
    else:
        print("   WARN: .next directory missing (run npm run build)")

    if (frontend_dir / ".env.production").exists():
        print("   PASS: Production env file found")
    else:
        print("   WARN: .env.production missing")

    # Test Backend
    print("\n2. Backend Tests:")
    backend_dir = Path("backend")

    if (backend_dir / "requirements.txt").exists():
        print("   PASS: requirements.txt found")
    else:
        print("   FAIL: requirements.txt missing")
        return False

    if (backend_dir / "app" / "main.py").exists():
        print("   PASS: main.py found")
    else:
        print("   FAIL: main.py missing")
        return False

    if (backend_dir / "app" / "services" / "cad" / "cad_service.py").exists():
        print("   PASS: CAD service found")
    else:
        print("   FAIL: CAD service missing")
        return False

    if (backend_dir / "app" / "services" / "nlp" / "nlp_service.py").exists():
        print("   PASS: NLP service found")
    else:
        print("   FAIL: NLP service missing")
        return False

    # Test Deployment Files
    print("\n3. Deployment Config Tests:")

    if Path("DEPLOYMENT_GUIDE.md").exists():
        print("   PASS: Deployment guide found")
    else:
        print("   WARN: Deployment guide missing")

    if (frontend_dir / "render-frontend.yaml").exists():
        print("   PASS: Frontend render config found")
    else:
        print("   WARN: Frontend render config missing")

    if (backend_dir / "render-backend.yaml").exists():
        print("   PASS: Backend render config found")
    else:
        print("   WARN: Backend render config missing")

    print("\n" + "=" * 40)
    print("RESULT: Ready for deployment!")
    print("\nNext steps:")
    print("1. Push code to GitHub")
    print("2. Create Render services")
    print("3. Set environment variables")
    print("4. Deploy and test")

    return True

if __name__ == "__main__":
    main()