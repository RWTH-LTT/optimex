export function Home() {
  return (
    <div className="space-y-8">
      <div className="space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Time-explicit Transition Pathway Optimization with optimex
        </h1>
        <p className="text-xl text-muted-foreground">
          A Python package for transition pathway optimization based on time-explicit Life Cycle Assessment (LCA)
        </p>
      </div>

      <div className="prose prose-neutral dark:prose-invert max-w-none">
        <p>
          <code className="text-sm">optimex</code> helps identify optimal process portfolios and deployment timing in systems 
          with multiple processes producing the same product, aiming to minimize dynamically accumulating environmental 
          impacts over time.
        </p>

        <p>
          <code className="text-sm">optimex</code> builds on top of the optimization framework{' '}
          <a href="https://github.com/Pyomo/pyomo" target="_blank" rel="noopener noreferrer">
            pyomo
          </a>{' '}
          and the LCA framework{' '}
          <a href="https://docs.brightway.dev/en/latest" target="_blank" rel="noopener noreferrer">
            Brightway
          </a>
          . If you are looking for a time-explicit LCA rather than an optimization tool, make sure to check out{' '}
          <a
            href="https://docs.brightway.dev/projects/bw-timex/en/latest/"
            target="_blank"
            rel="noopener noreferrer"
          >
            bw_timex
          </a>
          .
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <div className="rounded-lg border bg-card p-6">
          <div className="space-y-2">
            <h3 className="font-semibold">Life Cycle Timing</h3>
            <p className="text-sm text-muted-foreground">
              Environmental impacts are spread across a process's life cycle: construction happens first, 
              use comes later, and end-of-life impacts follow.
            </p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="space-y-2">
            <h3 className="font-semibold">Technology Evolution</h3>
            <p className="text-sm text-muted-foreground">
              Future technologies may become more sustainable, reducing the environmental impacts later 
              in the expansion period.
            </p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="space-y-2">
            <h3 className="font-semibold">Emission Accumulation</h3>
            <p className="text-sm text-muted-foreground">
              Most impacts arise from the accumulation of emissions. optimex retains the timing of emissions 
              during inventory calculations.
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-2xl font-bold tracking-tight">Support</h2>
        <p className="text-muted-foreground">
          If you have any questions or need help, do not hesitate to contact us:
        </p>
        <ul className="list-disc list-inside space-y-1 text-muted-foreground">
          <li>
            Jan Tautorus (
            <a href="mailto:jan.tautorus@rwth-aachen.de" className="underline">
              jan.tautorus@rwth-aachen.de
            </a>
            )
          </li>
          <li>
            Timo Diepers (
            <a href="mailto:timo.diepers@ltt.rwth-aachen.de" className="underline">
              timo.diepers@ltt.rwth-aachen.de
            </a>
            )
          </li>
        </ul>
      </div>
    </div>
  )
}
