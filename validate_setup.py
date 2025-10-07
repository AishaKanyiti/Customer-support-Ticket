#!/usr/bin/env python3
"""
Validation script to check if all required files and dependencies are present
before deploying the Streamlit app.
"""

import os
import sys

def check_file(filepath, file_type="file"):
    """Check if a file exists and is accessible"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {filepath} NOT FOUND")
        return False

def check_imports():
    """Check if all required packages can be imported"""
    required_packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib.pyplot"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy.sparse"),
        ("joblib", "joblib"),
    ]
    
    print("\n📦 Checking Python packages...")
    all_ok = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} NOT INSTALLED")
            all_ok = False
    return all_ok

def main():
    print("=" * 60)
    print("🔍 Customer Support AI Dashboard - Setup Validation")
    print("=" * 60)
    
    # Check required data files
    print("\n📊 Checking data files...")
    data_files = [
        "customer_support_tickets_cleaned.csv",
        "customer_support_tickets_with_urgent.csv"
    ]
    data_ok = all(check_file(f) for f in data_files if check_file(f) or True)
    
    # Check required model files
    print("\n🤖 Checking model files...")
    model_files = [
        "urgency_model_engineered.joblib",
        "urgency_vectorizer_engineered.joblib",
        "urgency_feature_cols.joblib",
    ]
    model_ok = all(check_file(f) for f in model_files)
    
    # Check optional model files
    print("\n📦 Checking optional files...")
    optional_files = [
        "rag_doc_ids.joblib",
        "rag_doc_tfidf.joblib",
        "rag_vectorizer.joblib",
        "Ticket_urgency_model.joblib",
        "vectorizer.joblib"
    ]
    for f in optional_files:
        check_file(f)
    
    # Check configuration files
    print("\n⚙️ Checking configuration files...")
    config_files = [
        "requirements.txt",
        "app.py",
        ".streamlit/config.toml",
        ".gitignore"
    ]
    config_ok = all(check_file(f) for f in config_files)
    
    # Check imports
    imports_ok = check_imports()
    
    # Validate app.py syntax
    print("\n🔧 Validating app.py syntax...")
    try:
        with open("app.py", "r") as f:
            compile(f.read(), "app.py", "exec")
        print("✅ app.py syntax is valid")
        syntax_ok = True
    except SyntaxError as e:
        print(f"❌ app.py has syntax errors: {e}")
        syntax_ok = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_checks = [
        ("Data files", check_file("customer_support_tickets_cleaned.csv")),
        ("Model files", model_ok),
        ("Configuration files", config_ok),
        ("Python packages", imports_ok),
        ("App syntax", syntax_ok)
    ]
    
    for check_name, status in all_checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
    
    print("=" * 60)
    
    if all(status for _, status in all_checks):
        print("\n🎉 All checks passed! Your app is ready for deployment.")
        print("\n🚀 To run locally:")
        print("   streamlit run app.py")
        print("\n☁️ To deploy to Streamlit Cloud:")
        print("   1. Push to GitHub")
        print("   2. Go to share.streamlit.io")
        print("   3. Connect your repository")
        return 0
    else:
        print("\n⚠️ Some checks failed. Please fix the issues above before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

