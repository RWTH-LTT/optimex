# optimex Documentation

This directory contains the documentation for optimex built with [Docus](https://docus.dev/), a documentation theme for [Nuxt](https://nuxt.com/).

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ (for API documentation generation)

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

Generate API documentation and build the static site:

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

The API documentation is automatically regenerated when running `npm run generate`.

## Project Structure

- `content/` - Documentation content in Markdown format
  - `8.api/` - Auto-generated API reference (do not edit manually)
- `public/` - Static assets (images, logos, etc.)
- `components/` - Vue components
- `scripts/` - Build scripts (e.g., API doc generator)
- `app.config.ts` - Docus configuration
- `nuxt.config.ts` - Nuxt configuration

## Writing Documentation

Documentation pages are written in Markdown with MDC (Markdown Components) syntax. See the [Docus documentation](https://docus.dev/) for more information.

---

For a detailed explanation of how things work, check out [Docus](https://docus.dev).
