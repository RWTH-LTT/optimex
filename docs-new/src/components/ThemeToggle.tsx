import { Moon, Sun } from 'lucide-react'
import { useTheme } from './ThemeProvider'

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  const toggleTheme = () => {
    if (theme === 'light') {
      setTheme('dark')
    } else if (theme === 'dark') {
      setTheme('system')
    } else {
      setTheme('light')
    }
  }

  const getIcon = () => {
    if (theme === 'dark') {
      return <Moon className="h-5 w-5" />
    } else if (theme === 'light') {
      return <Sun className="h-5 w-5" />
    } else {
      // System - show current system preference
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      return isDark ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />
    }
  }

  const getTitle = () => {
    if (theme === 'dark') return 'Dark mode'
    if (theme === 'light') return 'Light mode'
    return 'System theme'
  }

  return (
    <button
      onClick={toggleTheme}
      className="inline-flex items-center justify-center rounded-md font-medium transition-colors hover:bg-accent hover:text-accent-foreground h-9 w-9"
      title={getTitle()}
      aria-label={`Toggle theme (${getTitle()})`}
    >
      {getIcon()}
    </button>
  )
}
