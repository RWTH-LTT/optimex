"""Convert Jupyter notebooks in notebooks/ to Markdown for the docs.

For each notebook listed in NOTEBOOKS:
- Runs nbconvert (markdown output) into docs/content/examples/
- Strips ANSI escape sequences from every output cell
- Moves output images into a <stem>_files/ subdirectory
- Prepends YAML front-matter (icon + tags) required by Zensical

Run this script from the project root before building the docs:

    python docs/convert_notebooks.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
OUTPUT_DIR = REPO_ROOT / "docs" / "content" / "examples"

# Map notebook stem → (Zensical icon, list of tags)
NOTEBOOK_META: dict[str, tuple[str, list[str]]] = {
    "basic_example": (
        "lucide/play",
        ["learning", "tutorial"],
    ),
    "methanol_and_iron": (
        "lucide/flask-conical",
        ["case study", "industry"],
    ),
}

NOTEBOOK_SOURCE_PATHS: dict[str, str] = {
    "basic_example": "notebooks/basic_example.ipynb",
    "methanol_and_iron": "notebooks/methanol_and_iron.ipynb",
}

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def convert(
    notebook_path: Path, output_dir: Path, icon: str, tags: list[str]
) -> Path:
    """Convert *notebook_path* to Markdown in *output_dir*, strip ANSI codes,
    organise images into a <stem>_files/ sub-directory, and inject the Zensical
    icon and tags front-matter.  Returns the output file path."""
    try:
        from nbconvert.exporters import MarkdownExporter
    except ImportError:
        print(
            "nbconvert is not installed. "
            "Run: pip install nbconvert  (or add it to docs/requirements.txt)",
            file=sys.stderr,
        )
        sys.exit(1)

    stem = notebook_path.stem
    files_dir_name = f"{stem}_files"
    files_dir = output_dir / files_dir_name

    exporter = MarkdownExporter()
    body, resources = exporter.from_filename(str(notebook_path))

    # Strip ANSI codes
    body = strip_ansi(body)

    # Rewrite bare image refs  ![png](output_N_M.png)
    # → ![png](<stem>_files/output_N_M.png)
    body = re.sub(
        r"!\[([^\]]*)\]\((?!http)(output_[^)]+\.png)\)",
        rf"![\1]({files_dir_name}/\2)",
        body,
    )

    # Rewrite notebook-local data asset paths for rendered docs pages.
    # In notebooks, assets live at data/<file>; in the generated docs source,
    # they should remain relative to the examples section root. The builder
    # then rebases them correctly for /content/examples/<page>/ URLs.
    body = re.sub(r"src=(['\"])data/", r"src=\1data/", body)
    body = body.replace("](../data/", "](data/")

    # Prefer standard Markdown image syntax over raw HTML for notebook figures
    # so docs rendering stays consistent across builders.
    body = re.sub(
        r'<div style="display: flex; background-color: white; border-radius: 15px; '
        r'padding: 10px; width: 100%; max-width: 800px; margin: 0 auto;">\s*'
        r'<img src="(data/[^"]+)" style="border-radius: 15px; width: 100%;">\s*'
        r"</div>",
        r'![Product system flowchart](\1){ .example-flowchart }',
        body,
        flags=re.MULTILINE,
    )

    source_path = NOTEBOOK_SOURCE_PATHS.get(stem)
    source_override = ""
    if source_path:
        source_override = (
            "\n"
            f'<div hidden data-source-edit-path="{source_path}" '
            f'data-source-view-path="{source_path}"></div>\n'
        )

    # Build YAML front-matter lines
    tags_yaml = "\n".join(f"  - {t}" for t in tags)
    frontmatter = f"---\nicon: {icon}\ntags:\n{tags_yaml}\n---\n\n"
    body = frontmatter + source_override + body

    # Write markdown file
    md_path = output_dir / f"{stem}.md"
    md_path.write_text(body, encoding="utf-8")

    # Write image files into the _files/ subdirectory
    output_images = resources.get("outputs", {})
    if output_images:
        files_dir.mkdir(parents=True, exist_ok=True)
        for filename, data in output_images.items():
            (files_dir / filename).write_bytes(data)

    return md_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up any stray image files left over from a previous run that wrote
    # images directly into OUTPUT_DIR (old behaviour of FilesWriter)
    for leftover in OUTPUT_DIR.glob("output_*.png"):
        leftover.unlink()

    for stem, (icon, tags) in NOTEBOOK_META.items():
        notebook_path = NOTEBOOKS_DIR / f"{stem}.ipynb"
        if not notebook_path.exists():
            print(f"WARNING: notebook not found: {notebook_path}", file=sys.stderr)
            continue
        out = convert(notebook_path, OUTPUT_DIR, icon, tags)
        print(f"Converted {notebook_path.name} → {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
