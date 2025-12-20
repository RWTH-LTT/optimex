#!/usr/bin/env python3
"""
Convert Jupyter notebooks to HTML for embedding in the documentation site.
"""
import json
import os
import sys
from pathlib import Path

def convert_notebooks():
    """Convert notebooks to JSON format for React rendering."""
    # Get the repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    notebooks_dir = repo_root / "notebooks"
    output_dir = script_dir.parent / "public" / "notebooks"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not notebooks_dir.exists():
        print(f"Notebooks directory not found: {notebooks_dir}")
        return
    
    # List of notebooks to convert
    notebooks = [
        "basic_optimex_example.ipynb",
        "mini_hydrogen_case.ipynb" ,
        "h2.ipynb",
        "methanol.ipynb",
        "cdr.ipynb",
        "basic_optimex_example_two_decision_layers.ipynb"
    ]
    
    converted = []
    for notebook_name in notebooks:
        notebook_path = notebooks_dir / notebook_name
        if not notebook_path.exists():
            print(f"Skipping {notebook_name} (not found)")
            continue
            
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)
        
        # Save to public directory
        output_path = output_dir / notebook_name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f)
        
        print(f"Converted: {notebook_name}")
        converted.append(notebook_name)
    
    # Create index file
    index_path = output_dir / "index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({
            "notebooks": [{"name": nb, "path": f"/notebooks/{nb}"} for nb in converted]
        }, f, indent=2)
    
    print(f"\nConverted {len(converted)} notebooks")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    convert_notebooks()
