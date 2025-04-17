#!/usr/bin/env python
"""
Setup script for Deep Fake Detector project.
This script helps in setting up the environment and installing dependencies.
"""

import os
import subprocess
import sys
import platform

def main():
    print("Setting up Deep Fake Detector project environment...")
    
    # Determine OS
    system = platform.system()
    print(f"Detected OS: {system}")
    
    # Create virtual environment
    print("\nStep 1: Creating virtual environment...")
    if system == "Windows":
        subprocess.call([sys.executable, "-m", "venv", "venv"])
        activate_cmd = os.path.join("venv", "Scripts", "activate")
    else:  # Linux or MacOS
        subprocess.call([sys.executable, "-m", "venv", "venv"])
        activate_cmd = os.path.join("venv", "bin", "activate")
    
    print(f"Virtual environment created. To activate, run: {activate_cmd}")
    
    # Install dependencies
    print("\nStep 2: Installing dependencies...")
    print("To install dependencies, after activating the virtual environment, run:")
    if system == "Windows":
        print("pip install -r requirements.txt")
    else:
        print("pip install -r requirements.txt")
    
    print("\nStep 3: Verifying GPU support for TensorFlow (optional)...")
    print("After installing dependencies, you can check if TensorFlow detects your GPU:")
    print("python -c \"import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))\"")
    
    print("\nSetup instructions completed!")

if __name__ == "__main__":
    main()