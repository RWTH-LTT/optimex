import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ChevronRight, ExternalLink, AlertCircle } from 'lucide-react'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import { CopyButton } from '../components/CopyButton'

interface NotebookCell {
  cell_type: string
  source: string | string[]
  outputs?: any[]
  execution_count?: number | null
}

interface NotebookData {
  cells: NotebookCell[]
  metadata?: any
  nbformat?: number
  nbformat_minor?: number
}

function CodeCell({ cell }: { cell: NotebookCell }) {
  const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source

  useEffect(() => {
    Prism.highlightAll()
  }, [source])

  return (
    <div className="border rounded-lg overflow-hidden mb-4">
      {/* Input */}
      <div className="bg-muted/30 relative">
        <div className="flex items-center justify-between gap-2 px-4 py-2 border-b bg-muted/50">
          <span className="text-xs font-mono text-muted-foreground">
            [{cell.execution_count || ' '}]:
          </span>
          <CopyButton text={source} />
        </div>
        <pre className="p-4 overflow-x-auto !m-0 !bg-transparent">
          <code className="language-python text-sm">{source}</code>
        </pre>
      </div>

      {/* Outputs */}
      {cell.outputs && cell.outputs.length > 0 && (
        <div className="border-t">
          {cell.outputs.map((output, idx) => (
            <Output key={idx} output={output} />
          ))}
        </div>
      )}
    </div>
  )
}

function Output({ output }: { output: any }) {
  if (output.output_type === 'stream') {
    const text = Array.isArray(output.text) ? output.text.join('') : output.text
    return (
      <pre className="p-4 bg-background text-sm overflow-x-auto font-mono">
        {text}
      </pre>
    )
  }

  if (output.output_type === 'execute_result' || output.output_type === 'display_data') {
    // Handle different MIME types
    if (output.data) {
      if (output.data['text/html']) {
        const html = Array.isArray(output.data['text/html']) 
          ? output.data['text/html'].join('') 
          : output.data['text/html']
        return (
          <div 
            className="p-4 overflow-x-auto prose prose-sm dark:prose-invert max-w-none"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        )
      }
      
      if (output.data['image/png']) {
        return (
          <div className="p-4">
            <img 
              src={`data:image/png;base64,${output.data['image/png']}`} 
              alt="Output" 
              className="max-w-full h-auto"
            />
          </div>
        )
      }
      
      if (output.data['text/plain']) {
        const text = Array.isArray(output.data['text/plain']) 
          ? output.data['text/plain'].join('') 
          : output.data['text/plain']
        return (
          <pre className="p-4 bg-background text-sm overflow-x-auto font-mono">
            {text}
          </pre>
        )
      }
    }
  }

  if (output.output_type === 'error') {
    return (
      <div className="p-4 bg-destructive/10 border-l-4 border-destructive">
        <pre className="text-sm font-mono text-destructive overflow-x-auto">
          {output.ename}: {output.evalue}
        </pre>
      </div>
    )
  }

  return null
}

function MarkdownCell({ cell }: { cell: NotebookCell }) {
  const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source
  
  // Simple markdown rendering - convert common patterns
  const renderMarkdown = (text: string) => {
    return text
      .split('\n')
      .map((line, idx) => {
        // Headers
        if (line.startsWith('# ')) {
          return <h1 key={idx} className="text-3xl font-bold mt-6 mb-4">{line.substring(2)}</h1>
        }
        if (line.startsWith('## ')) {
          return <h2 key={idx} className="text-2xl font-bold mt-5 mb-3">{line.substring(3)}</h2>
        }
        if (line.startsWith('### ')) {
          return <h3 key={idx} className="text-xl font-bold mt-4 mb-2">{line.substring(4)}</h3>
        }
        
        // Code blocks
        if (line.startsWith('```')) {
          return <div key={idx} className="font-mono text-sm bg-muted p-2 rounded my-2">{line}</div>
        }
        
        // Bold and italic
        let processedLine = line
        processedLine = processedLine.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        processedLine = processedLine.replace(/\*(.+?)\*/g, '<em>$1</em>')
        processedLine = processedLine.replace(/`(.+?)`/g, '<code class="bg-muted px-1 py-0.5 rounded text-sm">$1</code>')
        
        if (line.trim() === '') {
          return <br key={idx} />
        }
        
        return <p key={idx} className="mb-2" dangerouslySetInnerHTML={{ __html: processedLine }} />
      })
  }

  return (
    <div className="prose prose-neutral dark:prose-invert max-w-none mb-4">
      {renderMarkdown(source)}
    </div>
  )
}

export function NotebookViewer() {
  const { notebook } = useParams<{ notebook: string }>()
  const [notebookData, setNotebookData] = useState<NotebookData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!notebook) return

    fetch(`/notebooks/${notebook}.ipynb`)
      .then((res) => {
        if (!res.ok) throw new Error('Notebook not found')
        return res.json()
      })
      .then((data) => {
        setNotebookData(data)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [notebook])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-muted-foreground">Loading notebook...</div>
      </div>
    )
  }

  if (error || !notebookData) {
    return (
      <div className="space-y-4">
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
          <div>
            <p className="font-semibold text-destructive">Error loading notebook</p>
            <p className="text-sm text-muted-foreground">{error || 'Notebook not found'}</p>
          </div>
        </div>
        <Link to="/examples" className="inline-flex items-center gap-1 text-sm text-primary hover:underline">
          ← Back to Examples
        </Link>
      </div>
    )
  }

  const notebookTitle = notebook
    ?.split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')

  return (
    <div className="space-y-6">
      <nav className="flex items-center gap-2 text-sm text-muted-foreground">
        <Link to="/examples" className="hover:text-primary transition-colors">
          Examples
        </Link>
        <ChevronRight className="h-4 w-4" />
        <span className="text-foreground font-medium">{notebookTitle}</span>
      </nav>

      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">{notebookTitle}</h1>
          <p className="text-muted-foreground mt-2">
            Interactive Jupyter notebook demonstrating optimex capabilities
          </p>
        </div>
        <div className="flex gap-2">
          <a
            href={`https://mybinder.org/v2/gh/RWTH-LTT/optimex/main?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2F${notebook}.ipynb`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-sm px-3 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            <ExternalLink className="h-4 w-4" />
            Launch on Binder
          </a>
          <a
            href={`https://github.com/RWTH-LTT/optimex/blob/main/notebooks/${notebook}.ipynb`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-sm px-3 py-2 rounded-md border hover:bg-accent transition-colors"
          >
            View on GitHub
          </a>
        </div>
      </div>

      <div className="border rounded-lg p-6 bg-card">
        {notebookData.cells.map((cell, idx) => {
          if (cell.cell_type === 'code') {
            return <CodeCell key={idx} cell={cell} />
          } else if (cell.cell_type === 'markdown') {
            return <MarkdownCell key={idx} cell={cell} />
          }
          return null
        })}
      </div>
    </div>
  )
}
