#!/usr/bin/env python3
"""
Dog Matchmaker Project Setup Script
Automatically creates the complete project structure

Run with: python setup_project.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_step(step_num, text):
    """Print a step with number."""
    print(f"\n[{step_num}] {text}")

def print_success(text):
    """Print success message."""
    print(f"  ✅ {text}")

def print_error(text):
    """Print error message."""
    print(f"  ❌ {text}")

def print_warning(text):
    """Print warning message."""
    print(f"  ⚠️  {text}")

def check_command(command):
    """Check if a command is available."""
    return shutil.which(command) is not None

def run_command(command, description, required=True):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print_success(description)
        return True
    except subprocess.CalledProcessError as e:
        if required:
            print_error(f"{description} failed: {e.stderr}")
            return False
        else:
            print_warning(f"{description} failed (optional)")
            return False

def create_directory_structure():
    """Create the project directory structure."""
    print_step(1, "Creating directory structure...")
    
    directories = [
        'data',
        'outputs',
        'Dog-Breeds-Dataset',
        'Dog-Breeds-Dataset/Images'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_success(f"Created: {directory}/")
    
    return True

def create_requirements_file():
    """Create requirements.txt file."""
    print_step(2, "Creating requirements.txt...")
    
    requirements = """streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
Pillow==10.0.0
jupyter==1.0.0
notebook==7.0.6
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print_success("Created requirements.txt")
    return True

def create_gitignore():
    """Create .gitignore file."""
    print_step(3, "Creating .gitignore...")
    
    gitignore = """# Virtual Environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Output files
outputs/*.png
outputs/*.jpg
outputs/*.json

# OS files
.DS_Store
Thumbs.db
*.swp
*.swo

# IDE
.vscode/
.idea/
*.sublime-*

# Streamlit
.streamlit/

# Data (keep directory but not large files)
Dog-Breeds-Dataset/*.zip
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore)
    
    print_success("Created .gitignore")
    return True

def create_readme():
    """Create a basic README.md."""
    print_step(4, "Creating README.md...")
    
    readme = """# 🐕 Dog Matchmaker AI

An intelligent AI-powered chatbot that recommends the top 3 dog breeds based on user lifestyle and personality preferences.

## 🚀 Quick Start

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Clone dog images dataset
git clone https://github.com/maartenvandenbroeck/Dog-Breeds-Dataset.git
```

### Run the App
```bash
streamlit run app.py
```

### Run the Analysis
```bash
jupyter notebook notebook.ipynb
```

## 📁 Project Structure

```
dog-matchmaker/
├── app.py                      # Streamlit chatbot
├── notebook.ipynb              # Analysis notebook
├── requirements.txt            # Dependencies
├── data/                       # CSV data files
├── outputs/                    # Generated visualizations
└── Dog-Breeds-Dataset/         # Breed images
```

## 📊 Features

- Natural language chatbot interface
- Data-driven breed matching algorithm
- Interactive visualizations
- Personalized recommendations with explanations
- Social media post generation

## 🏆 DataCamp Competition Submission

This project was created for the DataCamp AI-Powered Robot Dog Challenge.

---

**Built with ❤️ for dog lovers everywhere** 🐕🐾
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    print_success("Created README.md")
    return True

def create_data_readme():
    """Create README in data directory."""
    print_step(5, "Creating data/README.md...")
    
    data_readme = """# Data Directory

Place your CSV files here:

- `breed_traits.csv` - Contains breed characteristics (195 breeds × 17 columns)
- `trait_description.csv` - Contains trait descriptions (16 traits)

## Data Source

These files should be provided as part of the DataCamp competition dataset.

## Format

### breed_traits.csv
```
Breed,Affectionate With Family,Good With Young Children,...
Retrievers (Labrador),5,5,...
French Bulldogs,5,5,...
```

### trait_description.csv
```
Trait,Trait_1,Trait_5,Description
Affectionate With Family,Independent,Lovey-Dovey,...
```
"""
    
    with open('data/README.md', 'w') as f:
        f.write(data_readme)
    
    print_success("Created data/README.md")
    return True

def clone_dog_dataset():
    """Clone the Dog Breeds Dataset repository."""
    print_step(6, "Cloning Dog Breeds Dataset...")
    
    if os.path.exists('Dog-Breeds-Dataset/.git'):
        print_warning("Dataset already cloned, skipping...")
        return True
    
    if not check_command('git'):
        print_error("Git is not installed. Please install git to download images.")
        print_warning("You can manually download from: https://github.com/maartenvandenbroeck/Dog-Breeds-Dataset")
        return False
    
    success = run_command(
        'git clone https://github.com/maartenvandenbroeck/Dog-Breeds-Dataset.git',
        'Cloned Dog Breeds Dataset',
        required=False
    )
    
    if success:
        # Check if images exist
        images_path = Path('Dog-Breeds-Dataset/Images')
        if images_path.exists():
            image_count = len(list(images_path.glob('*.jpg'))) + len(list(images_path.glob('*.png')))
            print_success(f"Found {image_count} breed images")
    
    return True

def create_sample_app():
    """Create a minimal app.py file."""
    print_step(7, "Creating app.py placeholder...")
    
    if os.path.exists('app.py'):
        print_warning("app.py already exists, skipping...")
        return True
    
    app_code = """# Dog Matchmaker Streamlit App
# Replace this with the complete app code from the artifact

import streamlit as st

st.title("🐕 Dog Matchmaker AI")
st.write("Replace this file with the complete app.py code from the artifacts.")
st.info("Copy the full Streamlit app code here to get started!")
"""
    
    with open('app.py', 'w') as f:
        f.write(app_code)
    
    print_success("Created app.py placeholder")
    return True

def create_sample_notebook():
    """Create a minimal notebook.ipynb file."""
    print_step(8, "Creating notebook.ipynb placeholder...")
    
    if os.path.exists('notebook.ipynb'):
        print_warning("notebook.ipynb already exists, skipping...")
        return True
    
    notebook_json = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Dog Matchmaker Analysis Notebook\\n",
    "\\n",
    "Replace this with the complete notebook code from the artifact.\\n",
    "\\n",
    "Copy the full Jupyter notebook code here to get started!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Your analysis code here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    with open('notebook.ipynb', 'w') as f:
        f.write(notebook_json)
    
    print_success("Created notebook.ipynb placeholder")
    return True

def create_outputs_readme():
    """Create README in outputs directory."""
    print_step(9, "Creating outputs/README.md...")
    
    outputs_readme = """# Outputs Directory

This directory contains generated visualizations and results from the analysis notebook.

## Generated Files

When you run the notebook, the following files will be created:

### Visualizations
- `01_trait_distributions.png` - Histograms of key traits
- `02_correlation_heatmap.png` - Trait correlation matrix
- `03_top_family_breeds.png` - Family-friendly rankings
- `04_energy_vs_trainability.png` - Scatter plot analysis
- `05_shedding_vs_grooming.png` - Maintenance requirements
- `06_personality_clusters.png` - PCA analysis
- `07_lifestyle_recommendations.png` - Breed recommendations by lifestyle
- `08_trainable_breeds_radar.png` - Radar chart comparison
- `09_trait_variability.png` - Box plots
- `10_summary_statistics.png` - Statistics table

### Data Files
- `sample_recommendations.json` - Example recommendation outputs

All files are generated at 300 DPI for publication quality.
"""
    
    with open('outputs/README.md', 'w') as f:
        f.write(outputs_readme)
    
    print_success("Created outputs/README.md")
    return True

def check_data_files():
    """Check if data files exist."""
    print_step(10, "Checking for data files...")
    
    data_files = {
        'data/breed_traits.csv': 'Breed traits dataset',
        'data/trait_description.csv': 'Trait descriptions'
    }
    
    all_exist = True
    for filepath, description in data_files.items():
        if os.path.exists(filepath):
            print_success(f"Found: {description}")
        else:
            print_warning(f"Missing: {description}")
            print(f"           Please add {filepath}")
            all_exist = False
    
    return all_exist

def create_installation_guide():
    """Create INSTALL.md with detailed instructions."""
    print_step(11, "Creating INSTALL.md...")
    
    install_guide = """# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- git (optional, for cloning image dataset)

## Step-by-Step Installation

### 1. Verify Python Installation

```bash
python --version
# or
python3 --version
```

Should show Python 3.8 or higher.

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- streamlit (web app framework)
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (machine learning)
- matplotlib & seaborn (visualization)
- jupyter (notebook interface)

### 4. Add Data Files

Place these files in the `data/` directory:
- `breed_traits.csv`
- `trait_description.csv`

These should be provided as part of the DataCamp competition dataset.

### 5. Clone Image Dataset (Optional)

```bash
git clone https://github.com/maartenvandenbroeck/Dog-Breeds-Dataset.git
```

If you don't have git, download manually from:
https://github.com/maartenvandenbroeck/Dog-Breeds-Dataset

### 6. Add Application Code

Replace the placeholder files with the actual code:

**For app.py:**
Copy the complete Streamlit app code from the artifact.

**For notebook.ipynb:**
Copy the complete Jupyter notebook code from the artifact.

### 7. Test Installation

**Test Streamlit App:**
```bash
streamlit run app.py
```

**Test Notebook:**
```bash
jupyter notebook notebook.ipynb
```

## Troubleshooting

### "Command not found: streamlit"
Make sure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### "Module not found" errors
Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Images not showing
Ensure Dog-Breeds-Dataset is cloned in the project root directory.

### Port already in use (Streamlit)
Run on different port:
```bash
streamlit run app.py --server.port 8502
```

## Next Steps

1. ✅ Run the Jupyter notebook to generate visualizations
2. ✅ Launch the Streamlit app to test the chatbot
3. ✅ Review the generated outputs in `outputs/` directory
4. ✅ Customize as needed for your submission

## Need Help?

Check the main README.md or the documentation in each directory.
"""
    
    with open('INSTALL.md', 'w') as f:
        f.write(install_guide)
    
    print_success("Created INSTALL.md")
    return True

def print_summary():
    """Print setup summary and next steps."""
    print_header("Setup Complete! 🎉")
    
    print("\n📁 Project Structure Created:")
    print("  ✅ data/                    - Place CSV files here")
    print("  ✅ outputs/                 - Generated visualizations go here")
    print("  ✅ Dog-Breeds-Dataset/      - Breed images (if cloned)")
    print("  ✅ requirements.txt         - Python dependencies")
    print("  ✅ README.md                - Project documentation")
    print("  ✅ INSTALL.md               - Installation guide")
    print("  ✅ .gitignore               - Git ignore rules")
    print("  ✅ app.py (placeholder)     - Streamlit app")
    print("  ✅ notebook.ipynb (placeholder) - Analysis notebook")
    
    print("\n📋 Next Steps:")
    print("\n1. Add your data files to data/ directory:")
    print("   - breed_traits.csv")
    print("   - trait_description.csv")
    
    print("\n2. Replace placeholder code files:")
    print("   - Copy full app.py code from artifact")
    print("   - Copy full notebook.ipynb code from artifact")
    
    print("\n3. Install dependencies:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("   pip install -r requirements.txt")
    
    print("\n4. Run the application:")
    print("   streamlit run app.py")
    
    print("\n5. Or open the notebook:")
    print("   jupyter notebook notebook.ipynb")
    
    print("\n📚 Documentation:")
    print("   - README.md      - Project overview")
    print("   - INSTALL.md     - Detailed installation guide")
    print("   - data/README.md - Data file information")
    
    print("\n" + "=" * 60)
    print("  Ready to build your Dog Matchmaker! 🐕✨")
    print("=" * 60 + "\n")

def main():
    """Main setup function."""
    print_header("Dog Matchmaker Project Setup")
    print("This script will create the complete project structure\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required")
        sys.exit(1)
    
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Run setup steps
    steps = [
        create_directory_structure,
        create_requirements_file,
        create_gitignore,
        create_readme,
        create_data_readme,
        clone_dog_dataset,
        create_sample_app,
        create_sample_notebook,
        create_outputs_readme,
        check_data_files,
        create_installation_guide
    ]
    
    success_count = 0
    for step in steps:
        try:
            if step():
                success_count += 1
        except Exception as e:
            print_error(f"Error in {step.__name__}: {str(e)}")
    
    # Print summary
    print_summary()
    
    # Final status
    if success_count == len(steps):
        print("✨ All steps completed successfully!")
        return 0
    else:
        print(f"⚠️  Completed {success_count}/{len(steps)} steps")
        print("Please check warnings above and complete manual steps.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
