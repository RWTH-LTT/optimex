export function Theory() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Theory</h1>
        <p className="text-xl text-muted-foreground">
          Understanding time-explicit transition pathway optimization
        </p>
      </div>

      <div className="prose prose-neutral dark:prose-invert max-w-none">
        <h2>Overview</h2>
        <p>
          Like other transition pathway optimization tools, optimex identifies the optimal timing and 
          scale of process deployments to minimize environmental impacts over a transition period. 
        </p>
        
        <p>
          What sets optimex apart is its integration of three additional temporal considerations for 
          environmental impacts:
        </p>

        <h2>1. Timing within Process Life Cycles</h2>
        <p>
          Environmental impacts are spread across a process's life cycle: construction happens first, 
          use comes later, and end-of-life impacts follow. optimex captures this by distributing 
          process inputs and outputs over time.
        </p>

        <h2>2. Technology Evolution</h2>
        <p>
          Future technologies may become more sustainable, reducing the environmental impacts later 
          in the expansion period. optimex reflects this by allowing process inventories to evolve 
          over time.
        </p>

        <h2>3. Accumulation of Emissions and Impacts</h2>
        <p>
          Most impacts arise from the accumulation of emissions, but are typically modeled as discrete 
          and independent pulses. optimex retains the timing of emissions during inventory calculations 
          and applies dynamic characterization to account for impact accumulation.
        </p>

        <h2>Optimization Approach</h2>
        <p>
          During the transition pathway optimization, optimex simultaneously accounts for these temporal 
          considerations, identifying the environmentally optimal process deployment over the transition 
          period.
        </p>

        <p>
          The optimization is formulated as a linear programming problem using Pyomo, considering:
        </p>
        <ul>
          <li>Process deployment decisions over time</li>
          <li>Operation levels of deployed processes</li>
          <li>Demand fulfillment constraints</li>
          <li>Environmental impact minimization</li>
          <li>Process coupling and operational limits</li>
        </ul>

        <h2>Integration with Brightway</h2>
        <p>
          optimex builds on the Brightway LCA framework to:
        </p>
        <ul>
          <li>Access and process life cycle inventory data</li>
          <li>Apply characterization factors for impact assessment</li>
          <li>Enable time-explicit impact calculations</li>
          <li>Support dynamic characterization methods</li>
        </ul>
      </div>
    </div>
  )
}
