# optimex Documentation

Modern documentation site for optimex built with Vite, React, shadcn UI, and Tailwind CSS.

## Features

- 🚀 **Fast**: Built with Vite for lightning-fast development and builds
- ⚛️ **Modern**: Uses React with TypeScript for type safety
- 🎨 **Beautiful**: Styled with Tailwind CSS and shadcn/ui components
- 📱 **Responsive**: Works perfectly on desktop, tablet, and mobile
- 🔍 **Auto-generated API**: API documentation automatically generated from source code docstrings
- 🧭 **Great Navigation**: Header navigation and breadcrumbs using shadcn components
- 🎯 **Minimal Design**: Clean, Vercel-inspired aesthetic

## Development

### Prerequisites

- Node.js 20.x or later
- Python 3.9 or later (for API doc generation)

### Install Dependencies

```bash
npm install
```

### Development Server

```bash
npm run dev
```

This will start the Vite dev server at `http://localhost:5173/`

### Build

```bash
npm run build
```

This will:
1. Generate API documentation from Python source code
2. Build the TypeScript code
3. Create an optimized production build in the `dist/` directory

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
docs-new/
├── public/              # Static assets
│   └── api-docs.json   # Auto-generated API documentation
├── scripts/
│   └── generate-api-docs.py  # Python script to extract docstrings
├── src/
│   ├── components/     # React components
│   │   ├── ui/        # shadcn UI components
│   │   ├── Header.tsx
│   │   ├── Layout.tsx
│   │   └── BreadcrumbNav.tsx
│   ├── pages/         # Page components
│   │   ├── Home.tsx
│   │   ├── Installation.tsx
│   │   ├── Theory.tsx
│   │   ├── Examples.tsx
│   │   └── ApiReference.tsx
│   ├── lib/
│   │   └── utils.ts   # Utility functions
│   ├── App.tsx        # Main app component
│   ├── index.css      # Global styles
│   └── main.tsx       # Entry point
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## API Documentation Generation

The API documentation is automatically generated from the optimex Python package docstrings. The generation happens:

1. Before every build (via the `prebuild` script)
2. During the build process

To manually regenerate the API docs:

```bash
python3 scripts/generate-api-docs.py
```

The script extracts:
- Module docstrings
- Class definitions and docstrings
- Function/method signatures and docstrings
- Return type annotations
- Parameter lists

## Deployment

The site can be deployed to any static hosting service:

### Build for Production

```bash
npm run build
```

The `dist/` directory will contain all the static files ready for deployment.

### Deployment Options

1. **Vercel**: Connect your repository and Vercel will auto-deploy
2. **Netlify**: Drag and drop the `dist/` folder or connect via Git
3. **GitHub Pages**: Use GitHub Actions to build and deploy
4. **Any static host**: Upload the contents of `dist/`

## Customization

### Styling

The site uses Tailwind CSS with a custom theme. To customize colors, spacing, etc., edit `tailwind.config.js`.

### Navigation

To add or modify navigation items, edit the `navItems` array in `src/components/Header.tsx`.

### Pages

To add a new page:

1. Create a new component in `src/pages/`
2. Add a route in `src/App.tsx`
3. Add a navigation link in `src/components/Header.tsx`

## License

Same license as the optimex package (BSD 3-Clause).

