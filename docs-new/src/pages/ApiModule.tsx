import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Code2, ChevronRight, ExternalLink } from 'lucide-react'

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

function ViewSourceButton({ moduleName, line }: { moduleName: string; line: number }) {
  const githubUrl = `https://github.com/RWTH-LTT/optimex/blob/main/${moduleName.replace(/\./g, '/')}.py#L${line}`
  
  return (
    <a
      href={githubUrl}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-primary transition-colors"
      title="View source on GitHub"
    >
      <Code2 className="h-3 w-3" />
      <span>View Source</span>
      <ExternalLink className="h-3 w-3" />
    </a>
  )
}

function FunctionDoc({ func, moduleName }: { func: FunctionInfo; moduleName: string }) {
  return (
    <div id={func.name} className="scroll-mt-20 border rounded-lg p-4 mb-4">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <FunctionSignature func={func} />
          <Docstring text={func.docstring} />
        </div>
        <div className="flex flex-col items-end gap-2 ml-4">
          <div className="text-xs text-muted-foreground">
            Line {func.line}
          </div>
          <ViewSourceButton moduleName={moduleName} line={func.line} />
        </div>
      </div>
    </div>
  )
}

function ClassDoc({ classInfo, moduleName }: { classInfo: ClassInfo; moduleName: string }) {
  return (
    <div id={classInfo.name} className="scroll-mt-20 border rounded-lg p-6 mb-6">
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
        <div className="flex flex-col items-end gap-2">
          <div className="text-xs text-muted-foreground">
            Line {classInfo.line}
          </div>
          <ViewSourceButton moduleName={moduleName} line={classInfo.line} />
        </div>
      </div>

      {classInfo.methods.length > 0 && (
        <div className="mt-6">
          <h4 className="text-lg font-semibold mb-3">Methods</h4>
          <div className="space-y-3">
            {classInfo.methods.map((method, i) => (
              <div key={i} id={`${classInfo.name}.${method.name}`} className="scroll-mt-20 border-l-2 border-muted pl-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <FunctionSignature func={method} />
                    <Docstring text={method.docstring} />
                  </div>
                  <ViewSourceButton moduleName={moduleName} line={method.line} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function Sidebar({ moduleInfo }: { moduleInfo: ModuleInfo }) {
  return (
    <div className="sticky top-20 space-y-6 max-h-[calc(100vh-6rem)] overflow-y-auto">
      <div>
        <h3 className="font-semibold mb-2 text-sm text-muted-foreground uppercase">On This Page</h3>
        <div className="space-y-1">
          {moduleInfo.classes.length > 0 && (
            <div className="space-y-1">
              <div className="text-sm font-medium py-1">Classes</div>
              {moduleInfo.classes.map((cls) => (
                <div key={cls.name} className="space-y-1">
                  <a
                    href={`#${cls.name}`}
                    className="block text-sm text-muted-foreground hover:text-primary transition-colors py-1 pl-3 border-l-2 border-transparent hover:border-primary"
                  >
                    {cls.name}
                  </a>
                  {cls.methods.length > 0 && (
                    <div className="pl-3 space-y-1">
                      {cls.methods.map((method) => (
                        <a
                          key={method.name}
                          href={`#${cls.name}.${method.name}`}
                          className="block text-xs text-muted-foreground hover:text-primary transition-colors py-0.5 pl-3 border-l border-transparent hover:border-primary"
                        >
                          {method.name}()
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          {moduleInfo.functions.length > 0 && (
            <div className="space-y-1 mt-4">
              <div className="text-sm font-medium py-1">Functions</div>
              {moduleInfo.functions.map((func) => (
                <a
                  key={func.name}
                  href={`#${func.name}`}
                  className="block text-sm text-muted-foreground hover:text-primary transition-colors py-1 pl-3 border-l-2 border-transparent hover:border-primary"
                >
                  {func.name}()
                </a>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export function ApiModule() {
  const { module } = useParams<{ module: string }>()
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

  if (!apiDocs || !module) {
    return (
      <div className="text-muted-foreground">No API documentation available.</div>
    )
  }

  const fullModuleName = `optimex.${module}`
  const moduleInfo = apiDocs[fullModuleName]

  if (!moduleInfo) {
    return (
      <div className="space-y-4">
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
          <p className="text-destructive">Module "{fullModuleName}" not found.</p>
        </div>
        <Link to="/api" className="inline-flex items-center gap-1 text-sm text-primary hover:underline">
          ← Back to API Overview
        </Link>
      </div>
    )
  }

  return (
    <div className="flex gap-8">
      <div className="flex-1 min-w-0">
        <div className="space-y-4">
          <nav className="flex items-center gap-2 text-sm text-muted-foreground">
            <Link to="/api" className="hover:text-primary transition-colors">
              API Reference
            </Link>
            <ChevronRight className="h-4 w-4" />
            <span className="text-foreground font-medium">{module}</span>
          </nav>

          <div className="space-y-2">
            <h1 className="text-4xl font-bold tracking-tight font-mono">{fullModuleName}</h1>
            {moduleInfo.module_doc && (
              <div className="bg-muted/50 p-4 rounded-lg">
                <Docstring text={moduleInfo.module_doc} />
              </div>
            )}
          </div>

          {moduleInfo.classes.length > 0 && (
            <div className="mt-8">
              <h2 className="text-2xl font-bold mb-4">Classes</h2>
              {moduleInfo.classes.map((cls, i) => (
                <ClassDoc key={i} classInfo={cls} moduleName={fullModuleName} />
              ))}
            </div>
          )}

          {moduleInfo.functions.length > 0 && (
            <div className="mt-8">
              <h2 className="text-2xl font-bold mb-4">Functions</h2>
              {moduleInfo.functions.map((func, i) => (
                <FunctionDoc key={i} func={func} moduleName={fullModuleName} />
              ))}
            </div>
          )}
        </div>
      </div>

      <aside className="hidden lg:block w-64 flex-shrink-0">
        <Sidebar moduleInfo={moduleInfo} />
      </aside>
    </div>
  )
}
