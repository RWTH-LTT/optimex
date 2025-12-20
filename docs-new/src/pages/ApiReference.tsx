import { useEffect, useState } from 'react'

interface FunctionInfo {
  name: string
  args: string[]
  returns: string | null
  docstring: string
  line: number
  is_async: boolean
}

interface MethodInfo extends FunctionInfo {}

interface ClassInfo {
  name: string
  bases: string[]
  docstring: string
  methods: MethodInfo[]
  line: number
}

interface ModuleInfo {
  functions: FunctionInfo[]
  classes: ClassInfo[]
  module_doc: string
}

interface ApiDocs {
  [moduleName: string]: ModuleInfo
}

function Docstring({ text }: { text: string }) {
  if (!text) return null

  return (
    <div className="mt-3 text-sm text-muted-foreground whitespace-pre-wrap">
      {text}
    </div>
  )
}

function FunctionSignature({ func }: { func: FunctionInfo }) {
  return (
    <div className="font-mono text-sm">
      <span className="text-purple-600 dark:text-purple-400">
        {func.is_async ? 'async ' : ''}
      </span>
      <span className="text-blue-600 dark:text-blue-400">def</span>{' '}
      <span className="font-bold">{func.name}</span>
      <span className="text-muted-foreground">(</span>
      {func.args.map((arg, i) => (
        <span key={i}>
          <span className="text-orange-600 dark:text-orange-400">{arg}</span>
          {i < func.args.length - 1 && <span className="text-muted-foreground">, </span>}
        </span>
      ))}
      <span className="text-muted-foreground">)</span>
      {func.returns && (
        <>
          <span className="text-muted-foreground"> → </span>
          <span className="text-green-600 dark:text-green-400">{func.returns}</span>
        </>
      )}
    </div>
  )
}

function FunctionDoc({ func }: { func: FunctionInfo }) {
  return (
    <div className="border rounded-lg p-4 mb-4">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <FunctionSignature func={func} />
          <Docstring text={func.docstring} />
        </div>
        <div className="text-xs text-muted-foreground ml-4">
          Line {func.line}
        </div>
      </div>
    </div>
  )
}

function ClassDoc({ classInfo }: { classInfo: ClassInfo }) {
  return (
    <div className="border rounded-lg p-6 mb-6">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-xl font-bold font-mono">
            class {classInfo.name}
            {classInfo.bases.length > 0 && (
              <span className="text-muted-foreground text-base">
                ({classInfo.bases.join(', ')})
              </span>
            )}
          </h3>
          <Docstring text={classInfo.docstring} />
        </div>
        <div className="text-xs text-muted-foreground">
          Line {classInfo.line}
        </div>
      </div>

      {classInfo.methods.length > 0 && (
        <div className="mt-6">
          <h4 className="text-lg font-semibold mb-3">Methods</h4>
          <div className="space-y-3">
            {classInfo.methods.map((method, i) => (
              <div key={i} className="border-l-2 border-muted pl-4">
                <FunctionSignature func={method} />
                <Docstring text={method.docstring} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ModuleDoc({ moduleName, moduleInfo }: { moduleName: string; moduleInfo: ModuleInfo }) {
  return (
    <div className="mb-12">
      <div className="mb-6">
        <h2 className="text-2xl font-bold font-mono mb-2">{moduleName}</h2>
        {moduleInfo.module_doc && (
          <div className="bg-muted p-4 rounded-lg">
            <Docstring text={moduleInfo.module_doc} />
          </div>
        )}
      </div>

      {moduleInfo.classes.length > 0 && (
        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-4">Classes</h3>
          {moduleInfo.classes.map((cls, i) => (
            <ClassDoc key={i} classInfo={cls} />
          ))}
        </div>
      )}

      {moduleInfo.functions.length > 0 && (
        <div>
          <h3 className="text-xl font-semibold mb-4">Functions</h3>
          {moduleInfo.functions.map((func, i) => (
            <FunctionDoc key={i} func={func} />
          ))}
        </div>
      )}
    </div>
  )
}

export function ApiReference() {
  const [apiDocs, setApiDocs] = useState<ApiDocs | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api-docs.json')
      .then((res) => res.json())
      .then((data) => {
        setApiDocs(data)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
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

  if (error) {
    return (
      <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
        <p className="text-destructive">Error loading API documentation: {error}</p>
      </div>
    )
  }

  if (!apiDocs) {
    return (
      <div className="text-muted-foreground">No API documentation available.</div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">API Reference</h1>
        <p className="text-xl text-muted-foreground">
          Complete API documentation for the optimex package
        </p>
      </div>

      <div className="prose prose-neutral dark:prose-invert max-w-none">
        <p>
          This reference documentation is automatically generated from the source code docstrings.
          Click on any module below to see its classes and functions.
        </p>
      </div>

      <div className="space-y-8">
        {Object.entries(apiDocs)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([moduleName, moduleInfo]) => (
            <ModuleDoc key={moduleName} moduleName={moduleName} moduleInfo={moduleInfo} />
          ))}
      </div>
    </div>
  )
}
