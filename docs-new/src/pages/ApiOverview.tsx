import { Link } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { BookOpen, Code, FileText } from 'lucide-react'

interface ModuleInfo {
  functions: any[]
  classes: any[]
  module_doc: string
}

interface ApiDocs {
  [moduleName: string]: ModuleInfo
}

export function ApiOverview() {
  const [apiDocs, setApiDocs] = useState<ApiDocs | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api-docs.json')
      .then((res) => res.json())
      .then((data) => {
        setApiDocs(data)
        setLoading(false)
      })
      .catch(() => {
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-muted-foreground">Loading API documentation...</div>
      </div>
    )
  }

  if (!apiDocs) {
    return (
      <div className="text-muted-foreground">No API documentation available.</div>
    )
  }

  const modules = Object.entries(apiDocs).sort(([a], [b]) => a.localeCompare(b))

  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">API Reference</h1>
        <p className="text-xl text-muted-foreground">
          Complete API documentation for the optimex package
        </p>
      </div>

      <div className="prose prose-neutral dark:prose-invert max-w-none">
        <p>
          The optimex package provides a comprehensive framework for time-explicit transition pathway optimization
          based on Life Cycle Assessment (LCA). The API is organized into the following modules:
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {modules.map(([moduleName, moduleInfo]) => {
          const classCount = moduleInfo.classes.length
          const functionCount = moduleInfo.functions.length
          const shortName = moduleName.replace('optimex.', '')
          
          return (
            <Link
              key={moduleName}
              to={`/api/${shortName}`}
              className="group block border rounded-lg p-6 hover:border-primary transition-colors"
            >
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-primary/10 p-3">
                  <BookOpen className="h-6 w-6 text-primary" />
                </div>
                <div className="flex-1 space-y-2">
                  <h2 className="text-xl font-bold font-mono group-hover:text-primary transition-colors">
                    {moduleName}
                  </h2>
                  {moduleInfo.module_doc && (
                    <p className="text-sm text-muted-foreground line-clamp-3">
                      {moduleInfo.module_doc.split('\n')[0]}
                    </p>
                  )}
                  <div className="flex gap-4 text-sm text-muted-foreground">
                    {classCount > 0 && (
                      <div className="flex items-center gap-1">
                        <Code className="h-4 w-4" />
                        <span>{classCount} {classCount === 1 ? 'class' : 'classes'}</span>
                      </div>
                    )}
                    {functionCount > 0 && (
                      <div className="flex items-center gap-1">
                        <FileText className="h-4 w-4" />
                        <span>{functionCount} {functionCount === 1 ? 'function' : 'functions'}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
