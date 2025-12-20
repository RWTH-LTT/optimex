#!/usr/bin/env python3
"""
Generate API documentation JSON from optimex source code docstrings.

This script extracts docstrings, function signatures, and class information
from the optimex Python package and generates structured JSON for the React app.
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocstringExtractor(ast.NodeVisitor):
    """Extract documentation from Python AST."""

    def __init__(self):
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.current_module = ""

    def visit_Module(self, node: ast.Module) -> None:
        """Visit a module node."""
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function documentation."""
        if self.current_module not in self.modules:
            self.modules[self.current_module] = {
                "functions": [],
                "classes": [],
                "module_doc": "",
            }

        # Get function signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        # Get docstring
        docstring = ast.get_docstring(node) or ""

        # Get return annotation
        returns = ast.unparse(node.returns) if node.returns else None

        func_info = {
            "name": node.name,
            "args": args,
            "returns": returns,
            "docstring": docstring,
            "line": node.lineno,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }

        self.modules[self.current_module]["functions"].append(func_info)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class documentation."""
        if self.current_module not in self.modules:
            self.modules[self.current_module] = {
                "functions": [],
                "classes": [],
                "module_doc": "",
            }

        # Get class docstring
        docstring = ast.get_docstring(node) or ""

        # Get base classes
        bases = [ast.unparse(base) for base in node.bases]

        # Get methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in item.args.args]
                returns = ast.unparse(item.returns) if item.returns else None
                methods.append(
                    {
                        "name": item.name,
                        "args": args,
                        "returns": returns,
                        "docstring": ast.get_docstring(item) or "",
                        "line": item.lineno,
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                    }
                )

        class_info = {
            "name": node.name,
            "bases": bases,
            "docstring": docstring,
            "methods": methods,
            "line": node.lineno,
        }

        self.modules[self.current_module]["classes"].append(class_info)
        self.generic_visit(node)


def extract_docs_from_file(file_path: Path, module_name: str) -> Dict[str, Any]:
    """Extract documentation from a single Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
            return {}

    extractor = DocstringExtractor()
    extractor.current_module = module_name

    # Get module docstring
    module_doc = ast.get_docstring(tree) or ""

    extractor.visit(tree)

    if module_name in extractor.modules:
        extractor.modules[module_name]["module_doc"] = module_doc
        return extractor.modules[module_name]

    return {
        "functions": [],
        "classes": [],
        "module_doc": module_doc,
    }


def generate_api_docs(source_dir: Path, output_file: Path) -> None:
    """Generate API documentation JSON from source directory."""
    api_docs = {}

    # Walk through the optimex package
    for py_file in source_dir.rglob("*.py"):
        # Skip __pycache__, tests, etc.
        if any(
            part.startswith("__pycache__")
            or part.startswith("test")
            or part == "setup.py"
            for part in py_file.parts
        ):
            continue

        # Get relative module path
        rel_path = py_file.relative_to(source_dir.parent)
        module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")

        # Extract docs
        docs = extract_docs_from_file(py_file, module_name)
        if docs and (docs["functions"] or docs["classes"] or docs["module_doc"]):
            api_docs[module_name] = docs

    # Write to JSON file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(api_docs, f, indent=2)

    print(f"Generated API docs: {output_file}")
    print(f"Documented modules: {len(api_docs)}")


if __name__ == "__main__":
    # Paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    source_dir = repo_root / "optimex"
    output_file = script_dir.parent / "public" / "api-docs.json"

    print(f"Source directory: {source_dir}")
    print(f"Output file: {output_file}")

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    generate_api_docs(source_dir, output_file)
