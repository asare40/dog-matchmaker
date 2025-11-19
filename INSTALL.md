# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- git (optional, for cloning image dataset)

## Step-by-Step Installation

### 1. Verify Python Installation

```powershell
python --version
```

Should show Python 3.8 or higher.

### 2. Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
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

Place these files in the `data\` directory:
- `breed_traits.csv`
- `trait_description.csv`

These should be provided as part of the DataCamp competition dataset.

### 5. Clone Image Dataset (Optional)

```powershell
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
```powershell
streamlit run app.py
```

**Test Notebook:**
```powershell
jupyter notebook notebook.ipynb
```

## Troubleshooting

### "Command not found: streamlit"
Make sure virtual environment is activated and dependencies are installed:
```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

### "Module not found" errors
Reinstall dependencies:
```powershell
pip install --upgrade -r requirements.txt
```

### Images not showing
Ensure Dog-Breeds-Dataset is cloned in the project root directory.

### Port already in use (Streamlit)
Run on different port:
```powershell
streamlit run app.py --server.port 8502
```

## PowerShell Execution Policy

If you get an error about execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Next Steps

1. ? Run the Jupyter notebook to generate visualizations
2. ? Launch the Streamlit app to test the chatbot
3. ? Review the generated outputs in `outputs\` directory
4. ? Customize as needed for your submission

## Need Help?

Check the main README.md or the documentation in each directory.
