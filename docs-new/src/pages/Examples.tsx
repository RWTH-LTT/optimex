import { Link } from 'react-router-dom'
import { BookOpen, ExternalLink } from 'lucide-react'

const notebooks = [
  {
    id: 'basic_optimex_example',
    title: 'Basic optimex Example',
    description: 'A step-by-step introduction to optimex covering basic concepts and workflow.',
  },
  {
    id: 'h2',
    title: 'Hydrogen Production Optimization',
    description: 'Optimize hydrogen production pathways considering temporal LCA impacts.',
  },
  {
    id: 'methanol',
    title: 'Methanol Production Case Study',
    description: 'Demonstration of methanol production pathway optimization with time-explicit LCA.',
  },
  {
    id: 'cdr',
    title: 'Carbon Dioxide Removal (CDR)',
    description: 'Optimize carbon dioxide removal strategies over time.',
  },
  {
    id: 'basic_optimex_example_two_decision_layers',
    title: 'Two Decision Layers Example',
    description: 'Advanced example with multiple decision layers in the optimization.',
  },
]

export function Examples() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Examples</h1>
        <p className="text-xl text-muted-foreground">
          Learn optimex through interactive Jupyter notebooks
        </p>
      </div>

      <div className="prose prose-neutral dark:prose-invert max-w-none">
        <h2>Interactive Notebooks</h2>
        <p>
          We provide several Jupyter notebooks that demonstrate how to use optimex for different 
          optimization scenarios. You can view them directly here, launch them on Binder for interactive execution,
          or download them from GitHub.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {notebooks.map((notebook) => (
          <div key={notebook.id} className="rounded-lg border bg-card p-6">
            <div className="flex items-start gap-4">
              <div className="rounded-lg bg-primary/10 p-3">
                <BookOpen className="h-6 w-6 text-primary" />
              </div>
              <div className="flex-1 space-y-3">
                <div>
                  <h3 className="text-lg font-semibold">{notebook.title}</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    {notebook.description}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Link
                    to={`/examples/${notebook.id}`}
                    className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-9 px-3"
                  >
                    View Notebook
                  </Link>
                  <a
                    href={`https://mybinder.org/v2/gh/RWTH-LTT/optimex/main?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2F${notebook.id}.ipynb`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-9 px-3"
                  >
                    <ExternalLink className="h-3 w-3" />
                    Binder
                  </a>
                  <a
                    href={`https://github.com/RWTH-LTT/optimex/blob/main/notebooks/${notebook.id}.ipynb`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-9 px-3"
                  >
                    <ExternalLink className="h-3 w-3" />
                    GitHub
                  </a>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="prose prose-neutral dark:prose-invert max-w-none">
        <h2>Getting Started</h2>
        <p>
          We recommend starting with the Basic optimex Example to understand the core concepts:
        </p>
        <ol>
          <li>Setting up your LCA database with Brightway</li>
          <li>Defining processes and their temporal characteristics</li>
          <li>Configuring optimization parameters</li>
          <li>Running the optimization</li>
          <li>Analyzing and visualizing results</li>
        </ol>

        <h2>Running Locally</h2>
        <p>
          To run these examples locally, clone the repository and install the required dependencies:
        </p>
        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
          <code>{`git clone https://github.com/RWTH-LTT/optimex.git
cd optimex
pip install -e ".[dev]"
jupyter notebook notebooks/`}</code>
        </pre>
      </div>
    </div>
  )
}
