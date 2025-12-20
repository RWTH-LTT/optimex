# optimex Documentation

This directory contains the documentation for optimex built with [Docus](https://docus.dev/), a documentation theme for [Nuxt](https://nuxt.com/).

## Getting Started

### Prerequisites

- Node.js 18+ and npm

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

Generate the static site:

```bash
npm run generate
```

The output will be in `.output/public` directory.

Preview the production build:

```bash
npm run preview
```

## Project Structure

- `content/` - Documentation content in Markdown format
- `public/` - Static assets (images, logos, etc.)
- `components/` - Vue components
- `app.config.ts` - Docus configuration
- `nuxt.config.ts` - Nuxt configuration

## Writing Documentation

Documentation pages are written in Markdown with MDC (Markdown Components) syntax. See the [Docus documentation](https://docus.dev/) for more information.

---

For a detailed explanation of how things work, check out [Docus](https://docus.dev).
