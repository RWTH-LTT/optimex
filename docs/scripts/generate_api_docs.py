#!/usr/bin/env python3
"""
Generate API documentation from Python source code using AST parsing.
Outputs markdown files compatible with Nuxt Docus.
"""

import ast
from pathlib import Path
from typing import List, Tuple, Optional


def get_docstring(node) -> Optional[str]:
    """Extract docstring from an AST node."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and node.body:
        first_stmt = node.body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
            if isinstance(first_stmt.value.value, str):
                return first_stmt.value.value
    return None


def get_function_signature(node: ast.FunctionDef) -> str:
    """Generate function signature string from AST node."""
    args_list = []
    
    # Positional arguments
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args_list.append(arg_str)
    
    # *args
    if node.args.vararg:
        vararg_str = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
        args_list.append(vararg_str)
    
    # **kwargs
    if node.args.kwarg:
        kwarg_str = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
        args_list.append(kwarg_str)
    
    # Return annotation
    return_annotation = ""
    if node.returns:
        return_annotation = f" -> {ast.unparse(node.returns)}"
    
    return f"({', '.join(args_list)}){return_annotation}"


def format_docstring(docstring: str) -> str:
    """Format docstring for markdown output."""
    if not docstring:
        return ""
    
    lines = docstring.strip().split('\n')
    formatted_lines = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Detect code blocks with indentation or backticks
        if '```' in line:
            formatted_lines.append(line)
            in_code_block = not in_code_block
        elif stripped.startswith('>>>') or stripped.startswith('...'):
            if not in_code_block:
                formatted_lines.append('```python')
                in_code_block = True
            formatted_lines.append(line)
        elif in_code_block and (not stripped or not stripped.startswith(('>>>', '...'))):
            formatted_lines.append('```')
            in_code_block = False
            if stripped:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    if in_code_block:
        formatted_lines.append('```')
    
    return '\n'.join(formatted_lines)


def parse_module(source_path: Path) -> Tuple[Optional[str], List, List]:
    """
    Parse a Python module and extract documentation.
    
    Returns:
        Tuple of (module_docstring, functions_list, classes_list)
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Warning: Could not parse {source_path}: {e}")
        return None, [], []
    
    module_doc = get_docstring(tree)
    functions = []
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
            # Only include module-level functions
            if isinstance(node.parent if hasattr(node, 'parent') else None, ast.Module):
                sig = get_function_signature(node)
                doc = get_docstring(node)
                functions.append((node.name, sig, doc))
        
        elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
            doc = get_docstring(node)
            methods = []
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if not item.name.startswith('_') or item.name == '__init__':
                        method_sig = get_function_signature(item)
                        method_doc = get_docstring(item)
                        methods.append((item.name, method_sig, method_doc))
            
            classes.append((node.name, doc, methods))
    
    return module_doc, functions, classes


def generate_module_docs(module_path: Path) -> str:
    """Generate markdown documentation for a Python module."""
    module_name = module_path.stem
    
    if module_name == "__init__":
        return ""
    
    module_doc, functions, classes = parse_module(module_path)
    
    lines = [f"# {module_name}\n"]
    
    # Module docstring
    if module_doc:
        lines.append(format_docstring(module_doc))
        lines.append("\n")
    
    # Document functions
    if functions:
        lines.append("## Functions\n")
        for name, sig, doc in sorted(functions):
            lines.append(f"### `{name}{sig}`\n")
            if doc:
                lines.append(format_docstring(doc))
                lines.append("\n")
    
    # Document classes
    if classes:
        lines.append("## Classes\n")
        for name, doc, methods in sorted(classes):
            lines.append(f"### `{name}`\n")
            if doc:
                lines.append(format_docstring(doc))
                lines.append("\n")
            
            if methods:
                lines.append("#### Methods\n")
                for method_name, method_sig, method_doc in sorted(methods):
                    lines.append(f"##### `{method_name}{method_sig}`\n")
                    if method_doc:
                        lines.append(format_docstring(method_doc))
                        lines.append("\n")
    
    return '\n'.join(lines)


def main():
    """Generate API documentation for the optimex package."""
    # Paths - script is in docs/scripts, so parent.parent is repo root
    repo_root = Path(__file__).parent.parent.parent.resolve()
    optimex_dir = repo_root / "optimex"
    docs_content_dir = Path(__file__).parent.parent / "content"
    api_dir = docs_content_dir / "8.api"
    
    # Create API documentation directory
    api_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all Python modules
    modules = sorted([f for f in optimex_dir.glob("*.py") if f.stem != "__init__"])
    
    # Generate index page
    index_lines = [
        "---",
        "title: API Reference",
        "description: API documentation for optimex Python package",
        "---",
        "",
        "# API Reference",
        "",
        "This section contains the API documentation automatically generated from the Python source code docstrings.",
        "",
        "The optimex package provides tools for transition pathway optimization based on time-explicit Life Cycle Assessment (LCA).",
        "",
        "## Modules",
        ""
    ]
    
    module_descriptions = {
        "optimizer": "Optimization model construction and solving",
        "converter": "Data conversion and validation",
        "lca_processor": "Life Cycle Assessment processing",
        "postprocessing": "Result analysis and visualization"
    }
    
    for module_path in modules:
        module_name = module_path.stem
        desc = module_descriptions.get(module_name, module_path.name)
        index_lines.append(f"- **[{module_name}](/api/{module_name})** - {desc}")
    
    index_lines.append("")
    
    # Write index
    with open(api_dir / "0.index.md", "w") as f:
        f.write('\n'.join(index_lines))
    
    print(f"Generated API index at {api_dir / '0.index.md'}")
    
    # Generate documentation for each module
    for i, module_path in enumerate(modules, start=1):
        module_name = module_path.stem
        output_file = api_dir / f"{i}.{module_name}.md"
        
        content = [
            "---",
            f"title: {module_name}",
            f"description: API documentation for optimex.{module_name}",
            "---",
            ""
        ]
        
        module_docs = generate_module_docs(module_path)
        if module_docs:
            content.append(module_docs)
            
            with open(output_file, "w") as f:
                f.write('\n'.join(content))
            
            print(f"Generated {output_file}")
        else:
            print(f"Skipped {module_name} (no content)")
    
    print("\n✓ API documentation generation complete!")
    print(f"\nGenerated {len(modules)} module documentation files in {api_dir}")


if __name__ == "__main__":
    main()
