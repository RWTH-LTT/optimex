import { useEffect } from 'react'
import Prism from 'prismjs'
import 'prismjs/components/prism-bash'
import { CopyButton } from '../components/CopyButton'

export function Installation() {
  useEffect(() => {
    Prism.highlightAll()
  }, [])

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Installation</h1>
        <p className="text-xl text-muted-foreground">
          Get started with optimex by installing it via pip or conda
        </p>
      </div>

      <div className="prose prose-neutral dark:prose-invert max-w-none">
        <h2>Install via pip</h2>
        <p>The easiest way to install optimex is using pip:</p>
        <div className="relative">
          <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
            <code className="language-bash">pip install optimex</code>
          </pre>
          <div className="absolute top-2 right-2">
            <CopyButton text="pip install optimex" />
          </div>
        </div>

        <h2>Install via conda</h2>
        <p>You can also install optimex using conda:</p>
        <div className="relative">
          <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
            <code className="language-bash">conda install -c diepers optimex</code>
          </pre>
          <div className="absolute top-2 right-2">
            <CopyButton text="conda install -c diepers optimex" />
          </div>
        </div>

        <h2>Development Installation</h2>
        <p>
          To install optimex for development with all dev dependencies:
        </p>
        <div className="relative">
          <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
            <code className="language-bash">{`git clone https://github.com/RWTH-LTT/optimex.git
cd optimex
pip install -e ".[dev]"`}</code>
          </pre>
          <div className="absolute top-2 right-2">
            <CopyButton text={`git clone https://github.com/RWTH-LTT/optimex.git
cd optimex
pip install -e ".[dev]"`} />
          </div>
        </div>

        <h2>Requirements</h2>
        <p>optimex requires Python 3.9 or higher and has the following dependencies:</p>
        <ul>
          <li>dynamic-characterization</li>
          <li>brightway25</li>
          <li>pyomo</li>
          <li>pyyaml</li>
          <li>bw-temporalis</li>
        </ul>

        <h2>Verify Installation</h2>
        <p>You can verify your installation by importing optimex in Python:</p>
        <div className="relative">
          <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
            <code className="language-bash">{`python -c "import optimex; print(optimex.__version__)"`}</code>
          </pre>
          <div className="absolute top-2 right-2">
            <CopyButton text={`python -c "import optimex; print(optimex.__version__)"`} />
          </div>
        </div>
      </div>
    </div>
  )
}
