#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Docus-compatible markdown format.
Outputs markdown files with proper code highlighting and metadata.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


def clean_output(output: str) -> str:
    """Clean notebook output for better display."""
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', output)
    
    # Limit very long outputs
    lines = cleaned.split('\n')
    if len(lines) > 50:
        lines = lines[:25] + ['...', '(output truncated)', '...'] + lines[-25:]
    
    return '\n'.join(lines)


def convert_cell(cell: Dict[str, Any], cell_index: int) -> str:
    """Convert a single notebook cell to markdown."""
    cell_type = cell.get('cell_type')
    
    if cell_type == 'markdown':
        # Return markdown content as-is
        source = ''.join(cell.get('source', []))
        return source + '\n\n'
    
    elif cell_type == 'code':
        # Format code cell with syntax highlighting
        source = ''.join(cell.get('source', []))
        
        if not source.strip():
            return ''
        
        output_md = f'```python\n{source}\n```\n\n'
        
        # Add outputs if they exist
        outputs = cell.get('outputs', [])
        if outputs:
            for output in outputs:
                output_type = output.get('output_type')
                
                if output_type == 'stream':
                    # Text output from print statements
                    text = ''.join(output.get('text', []))
                    if text.strip():
                        cleaned_text = clean_output(text)
                        output_md += f'```plaintext\n{cleaned_text}\n```\n\n'
                
                elif output_type == 'execute_result' or output_type == 'display_data':
                    # Check for text representation
                    data = output.get('data', {})
                    
                    if 'text/plain' in data:
                        text = ''.join(data['text/plain'])
                        if text.strip():
                            cleaned_text = clean_output(text)
                            output_md += f'```plaintext\n{cleaned_text}\n```\n\n'
                    
                    # Check for images (base64 encoded)
                    if 'image/png' in data:
                        output_md += '::alert{type="info"}\n'
                        output_md += '*Note: Image output from this cell is not displayed in the documentation.*\n'
                        output_md += '::\n\n'
                
                elif output_type == 'error':
                    # Error output
                    traceback = '\n'.join(output.get('traceback', []))
                    if traceback.strip():
                        cleaned_tb = clean_output(traceback)
                        output_md += f'::alert{{type="error"}}\n```\n{cleaned_tb}\n```\n::\n\n'
        
        return output_md
    
    return ''


def convert_notebook(notebook_path: Path, output_path: Path, title: str = None):
    """Convert a Jupyter notebook to Docus markdown format."""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Get notebook title from first markdown cell or filename
    if not title:
        cells = notebook.get('cells', [])
        if cells and cells[0].get('cell_type') == 'markdown':
            first_line = ''.join(cells[0].get('source', [])).split('\n')[0]
            # Extract title from markdown heading
            title_match = re.match(r'^#+\s*(.+)', first_line)
            if title_match:
                title = title_match.group(1).strip()
        
        if not title:
            title = notebook_path.stem.replace('_', ' ').title()
    
    # Start building markdown content
    content = [
        '---',
        f'title: {title}',
        f'description: Jupyter notebook example - {title}',
        '---',
        '',
        f'# {title}',
        '',
        '::alert{type="info"}',
        'This page is automatically generated from a Jupyter notebook. You can [download the original notebook](https://github.com/RWTH-LTT/optimex/blob/main/notebooks/' + notebook_path.name + ') to run it yourself.',
        '::',
        ''
    ]
    
    # Convert cells
    cells = notebook.get('cells', [])
    for idx, cell in enumerate(cells):
        # Skip first cell if it was used for title
        if idx == 0 and cells[0].get('cell_type') == 'markdown':
            source = ''.join(cells[0].get('source', []))
            if source.strip().startswith('#'):
                continue
        
        cell_md = convert_cell(cell, idx)
        if cell_md:
            content.append(cell_md)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))


def main():
    """Convert all notebooks in the notebooks directory."""
    # Paths
    repo_root = Path(__file__).parent.parent.parent.resolve()
    notebooks_dir = repo_root / "notebooks"
    docs_content_dir = Path(__file__).parent.parent / "content"
    examples_dir = docs_content_dir / "9.notebooks"
    
    # Create notebooks documentation directory
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all notebooks
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    
    # Notebook titles mapping
    notebook_titles = {
        "basic_optimex_example.ipynb": "Basic optimex Example",
        "basic_optimex_example_two_decision_layers.ipynb": "Two Decision Layers Example",
        "h2.ipynb": "Hydrogen Production Pathway",
        "cdr.ipynb": "Carbon Dioxide Removal",
        "methanol.ipynb": "Methanol Production"
    }
    
    # Generate index page
    index_lines = [
        '---',
        'title: Examples & Notebooks',
        'description: Interactive Jupyter notebook examples for optimex',
        '---',
        '',
        '# Examples & Notebooks',
        '',
        'This section contains interactive Jupyter notebook examples demonstrating various use cases of optimex.',
        '',
        'All notebooks are available in the [GitHub repository](https://github.com/RWTH-LTT/optimex/tree/main/notebooks) and can be run locally or in [Binder](https://mybinder.org/v2/gh/RWTH-LTT/optimex/main?urlpath=%2Fdoc%2Ftree%2Fnotebooks).',
        '',
        '## Available Notebooks',
        ''
    ]
    
    for notebook_path in notebooks:
        notebook_name = notebook_path.name
        title = notebook_titles.get(notebook_name, notebook_path.stem.replace('_', ' ').title())
        index_lines.append(f'- **[{title}](/notebooks/{notebook_path.stem})** - {notebook_name}')
    
    index_lines.extend([
        '',
        '## Running Notebooks',
        '',
        '### Locally',
        '',
        '1. Clone the repository:',
        '```bash',
        'git clone https://github.com/RWTH-LTT/optimex.git',
        'cd optimex',
        '```',
        '',
        '2. Install dependencies:',
        '```bash',
        'conda create -n optimex -c conda-forge -c cmutel -c diepers optimex jupyterlab',
        'conda activate optimex',
        '```',
        '',
        '3. Launch Jupyter:',
        '```bash',
        'jupyter lab notebooks/',
        '```',
        '',
        '### Online (Binder)',
        '',
        'Click the badge below to launch an interactive environment:',
        '',
        '[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RWTH-LTT/optimex/main?urlpath=%2Fdoc%2Ftree%2Fnotebooks)',
        ''
    ])
    
    # Write index
    with open(examples_dir / "0.index.md", "w") as f:
        f.write('\n'.join(index_lines))
    
    print(f"Generated notebooks index at {examples_dir / '0.index.md'}")
    
    # Convert each notebook
    for i, notebook_path in enumerate(notebooks, start=1):
        notebook_name = notebook_path.name
        title = notebook_titles.get(notebook_name)
        output_file = examples_dir / f"{i}.{notebook_path.stem}.md"
        
        try:
            convert_notebook(notebook_path, output_file, title)
            print(f"Converted {notebook_name} -> {output_file}")
        except Exception as e:
            print(f"Error converting {notebook_name}: {e}")
    
    print(f"\n✓ Notebook conversion complete!")
    print(f"Generated {len(notebooks)} notebook documentation files in {examples_dir}")


if __name__ == "__main__":
    main()
