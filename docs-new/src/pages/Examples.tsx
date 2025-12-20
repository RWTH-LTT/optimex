export function Examples() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Examples</h1>
        <p className="text-xl text-muted-foreground">
          Learn optimex through practical examples
        </p>
      </div>

      <div className="prose prose-neutral dark:prose-invert max-w-none">
        <h2>Interactive Notebooks</h2>
        <p>
          We provide several Jupyter notebooks that demonstrate how to use optimex for different 
          optimization scenarios.
        </p>

        <div className="not-prose space-y-4 my-6">
          <div className="rounded-lg border bg-card p-6">
            <h3 className="text-lg font-semibold mb-2">Basic optimex Example</h3>
            <p className="text-sm text-muted-foreground mb-4">
              A step-by-step introduction to optimex covering basic concepts and workflow.
            </p>
            <div className="flex gap-2">
              <a
                href="https://mybinder.org/v2/gh/RWTH-LTT/optimex/main?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fbasic_optimex_example.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
              >
                Launch on Binder
              </a>
              <a
                href="https://github.com/RWTH-LTT/optimex/blob/main/notebooks/basic_optimex_example.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-10 px-4 py-2"
              >
                View on GitHub
              </a>
            </div>
          </div>

          <div className="rounded-lg border bg-card p-6">
            <h3 className="text-lg font-semibold mb-2">Mini Hydrogen Case Study</h3>
            <p className="text-sm text-muted-foreground mb-4">
              A practical example demonstrating optimization of hydrogen production pathways.
            </p>
            <div className="flex gap-2">
              <a
                href="https://mybinder.org/v2/gh/RWTH-LTT/optimex/main?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fmini_hydrogen_case.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
              >
                Launch on Binder
              </a>
              <a
                href="https://github.com/RWTH-LTT/optimex/blob/main/notebooks/mini_hydrogen_case.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-10 px-4 py-2"
              >
                View on GitHub
              </a>
            </div>
          </div>
        </div>

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
