"""
Basic pytest tests for PCOS Prediction project
Simple tests that verify basic functionality
"""
import os
import json

def test_project_structure():
    """Test that required project files exist"""
    assert os.path.exists("app/main.py")
    assert os.path.exists("requirements.txt")
    assert os.path.exists("params.yaml")
    assert os.path.exists("dvc.yaml")

def test_requirements_file():
    """Test that requirements.txt contains necessary packages"""
    with open("requirements.txt", "r") as f:
        content = f.read()
    
    required_packages = ["fastapi", "uvicorn", "mlflow", "dvc", "evidently", "pytest"]
    for package in required_packages:
        assert package in content, f"Missing required package: {package}"

def test_params_file():
    """Test that params.yaml exists and is valid"""
    assert os.path.exists("params.yaml")
    # Could add YAML parsing test here if needed

def test_models_directory():
    """Test that models directory structure exists"""
    assert os.path.exists("models") or os.path.exists("app/models")

def test_data_directory():
    """Test that data directory exists"""
    assert os.path.exists("data")
