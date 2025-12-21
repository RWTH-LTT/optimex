# optimex Documentation

This directory contains the documentation for optimex built with [Docus](https://docus.dev/), a documentation theme for [Nuxt](https://nuxt.com/).

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ (for API documentation and notebook conversion)

### Development

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The documentation will be available at http://localhost:3000

### Build for Production

Generate API documentation, convert notebooks, and build the static site:

```bash
npm run generate
```

The output will be in `.output/public` directory.

Preview the production build:

```bash
npm run preview
```

## API Documentation

The API Reference section is automatically generated from Python source code docstrings using AST parsing.

### Generate API Docs Manually

To regenerate the API documentation:

```bash
npm run api-docs
```

This runs `scripts/generate_api_docs.py` which:
- Parses Python modules in the `optimex` package
- Extracts docstrings from modules, classes, and functions
- Generates markdown files in `content/8.api/`
- Creates an index page listing all modules

## Jupyter Notebook Examples

The Examples & Notebooks section is automatically generated from Jupyter notebooks in the repository.

### Convert Notebooks Manually

To regenerate the notebook documentation:

```bash
npm run notebooks
```

This runs `scripts/convert_notebooks.py` which:
- Reads Jupyter notebooks from the `notebooks/` directory
- Converts cells to Docus-compatible markdown
- Applies syntax highlighting to code blocks
- Generates markdown files in `content/9.notebooks/`
- Creates an index page with links to all notebooks

Features:
- **Code highlighting**: All Python code blocks have syntax highlighting
- **Output preservation**: Cell outputs are preserved as plaintext blocks
- **Error handling**: Notebook errors are displayed in alert boxes
- **Download links**: Each notebook page includes a link to the original `.ipynb` file

The notebook conversion is automatically run when running `npm run generate`.

## Project Structure

- `content/` - Documentation content in Markdown format
  - `8.api/` - Auto-generated API reference (do not edit manually)
  - `9.notebooks/` - Auto-generated notebook examples (do not edit manually)
- `public/` - Static assets (images, logos, etc.)
- `components/` - Vue components
- `scripts/` - Build scripts
  - `generate_api_docs.py` - API doc generator
  - `convert_notebooks.py` - Notebook converter
- `app.config.ts` - Docus configuration
- `nuxt.config.ts` - Nuxt configuration

## Writing Documentation

Documentation pages are written in Markdown with MDC (Markdown Components) syntax. See the [Docus documentation](https://docus.dev/) for more information.

### Code Highlighting

Docus automatically provides syntax highlighting for code blocks. Simply use triple backticks with a language identifier:

\`\`\`python
import optimex
\`\`\`

Supported languages include: python, javascript, typescript, bash, json, yaml, and many more.

---

For a detailed explanation of how things work, check out [Docus](https://docus.dev).
