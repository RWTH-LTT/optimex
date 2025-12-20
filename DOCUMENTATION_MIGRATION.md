# Documentation Migration Guide

This guide explains how to transition from the old Sphinx-based documentation to the new Vite + React documentation.

## What's New

The documentation has been completely rewritten using modern web technologies:

- **Old**: Sphinx + MyST + pydata-sphinx-theme
- **New**: Vite + React + Tailwind CSS + shadcn/ui

## Benefits of the New Approach

1. **Faster builds**: Vite builds in seconds instead of minutes
2. **Better UX**: React provides smooth navigation and better interactivity
3. **Modern design**: Clean, minimal Vercel-inspired aesthetic
4. **Easier maintenance**: No complex Sphinx extensions or configuration
5. **Better mobile experience**: Fully responsive design
6. **Auto-generated API docs**: Directly from Python source code

## Directory Structure

```
Old (docs/):                    New (docs-new/):
├── conf.py                     ├── vite.config.ts
├── index.md                    ├── src/
├── content/                    │   ├── pages/
│   ├── installation.md         │   │   ├── Home.tsx
│   ├── theory.md               │   │   ├── Installation.tsx
│   ├── examples/               │   │   ├── Theory.tsx
│   └── api/                    │   │   ├── Examples.tsx
└── _static/                    │   │   └── ApiReference.tsx
                                │   └── components/
                                ├── scripts/
                                │   └── generate-api-docs.py
                                └── public/
                                    └── api-docs.json
```

## Content Migration Status

| Old Location | New Location | Status |
|--------------|-------------|---------|
| `index.md` | `src/pages/Home.tsx` | ✅ Migrated |
| `content/installation.md` | `src/pages/Installation.tsx` | ✅ Migrated |
| `content/theory.md` | `src/pages/Theory.tsx` | ✅ Migrated |
| `content/examples/` | `src/pages/Examples.tsx` | ✅ Migrated (links to notebooks) |
| `content/api/` (autoapi) | `src/pages/ApiReference.tsx` | ✅ Auto-generated from source |
| `content/contributing.md` | - | ⚠️ Not yet migrated |
| `content/changelog.md` | - | ⚠️ Not yet migrated |
| `content/license.md` | - | ⚠️ Not yet migrated |
| `content/codeofconduct.md` | - | ⚠️ Not yet migrated |

## Adding New Pages

To add a new page to the documentation:

1. Create a new component in `docs-new/src/pages/YourPage.tsx`:

```tsx
export function YourPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-4xl font-bold">Your Page Title</h1>
      <p>Your content here...</p>
    </div>
  )
}
```

2. Add a route in `docs-new/src/App.tsx`:

```tsx
import { YourPage } from './pages/YourPage'

// In the Routes component:
<Route path="your-page" element={<YourPage />} />
```

3. Add a navigation link in `docs-new/src/components/Header.tsx`:

```tsx
const navItems = [
  // ... existing items
  { label: 'Your Page', href: '/your-page' },
]
```

## Updating API Documentation

The API documentation is automatically generated from Python docstrings. To update it:

1. Edit the docstrings in the Python source code (`optimex/*.py`)
2. Run the generation script:
   ```bash
   cd docs-new
   python3 scripts/generate-api-docs.py
   ```
3. The changes will be reflected in the API reference page

### Docstring Format

The generator supports NumPy-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Short description of the function.
    
    Longer description with more details about what the function does,
    its behavior, and any important notes.
    
    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int
        Description of param2
        
    Returns
    -------
    bool
        Description of return value
    """
    pass
```

## Local Development

### Old Sphinx docs:
```bash
cd docs
conda env create -f environment.yaml
conda activate optimex_documentation
make html
```

### New Vite docs:
```bash
cd docs-new
npm install
npm run dev
```

The new approach is much simpler and faster!

## Building for Production

### Old:
```bash
cd docs
make clean html
# Output in docs/_build/html/
```

### New:
```bash
cd docs-new
npm run build
# Output in docs-new/dist/
```

## Deployment

### Old (ReadTheDocs):
- Configured via `.readthedocs.yml`
- Uses conda environment
- Builds on RTD servers

### New Options:

1. **Vercel** (Recommended):
   - Automatic deploys on git push
   - Global CDN
   - Free for open source
   - Config: `docs-new/vercel.json`

2. **Netlify**:
   - Drag & drop or Git integration
   - Config: `docs-new/netlify.toml`

3. **GitHub Pages**:
   - Build and deploy from Actions
   - Free for public repos

4. **Self-hosted**:
   - Just upload `dist/` folder to any web server

## Migration Checklist

If you want to fully migrate to the new docs:

- [ ] Review all migrated content for accuracy
- [ ] Migrate remaining pages (Contributing, Changelog, License, Code of Conduct)
- [ ] Update links in README.md to point to new docs
- [ ] Setup deployment (Vercel/Netlify/GitHub Pages)
- [ ] Test all links and navigation
- [ ] Archive old Sphinx docs or keep for reference
- [ ] Update `.readthedocs.yml` or switch to new platform

## Keeping Both (Transition Period)

You can keep both documentation systems during a transition period:

- Old docs at: `https://optimex.readthedocs.io/`
- New docs at: `https://optimex-docs.vercel.app/` (or similar)

Add a banner to the old docs pointing users to the new docs.

## Questions?

For questions about the new documentation:
1. Check the `docs-new/README.md` file
2. Review the source code in `docs-new/src/`
3. Contact the maintainers

## Rollback Plan

If you need to rollback to the old docs:

1. The old Sphinx docs are untouched in the `docs/` directory
2. Simply don't deploy the new `docs-new/` folder
3. Continue using ReadTheDocs as before

The new docs are completely separate and don't affect the old system.
