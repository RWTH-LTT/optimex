import os
import sys
import subprocess
import urllib.request
import json
from pathlib import Path

# Configuration
NB_URL = "https://raw.githubusercontent.com/RWTH-LTT/optimex/refs/heads/main/notebooks/basic_example.ipynb"
DEPS = ["optimex"]

def main():
    # 1. Handle Project Name Input
    # Usage: uv run script.py [project_name]
    project_name = sys.argv[1] if len(sys.argv) > 1 else "."
    project_path = Path(project_name).resolve()

    if project_name != ".":
        print(f"Creating project directory: {project_name}")
        project_path.mkdir(parents=True, exist_ok=True)
    
    # Change working directory to the new project path
    os.chdir(project_path)

    # 2. Initialize uv project
    if not (project_path / "pyproject.toml").exists():
        subprocess.run(["uv", "init", "--name", project_path.name], check=True)
        # Remove the boilerplate main.py
        hello_file = project_path / "main.py"
        if hello_file.exists():
            hello_file.unlink()

    # 3. Add dependencies
    print(f"Installing dependencies: {', '.join(DEPS)}...")
    subprocess.run(["uv", "add"] + DEPS, check=True)

    # 4. Download the notebook
    print("Fetching example notebook...")
    urllib.request.urlretrieve(NB_URL, "start_here.ipynb")

    print(f"\n✅ Done! Project '{project_path.name}' is ready.\n")
    print(f"Open in Jupyter Lab: cd {project_name} && uv run --with jupyter jupyter lab")
    print(f"Open in VS Code: cd {project_name} && code .")

if __name__ == "__main__":
    main()