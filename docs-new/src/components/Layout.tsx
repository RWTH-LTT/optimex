import { Outlet } from 'react-router-dom'
import { Header } from './Header'
import { BreadcrumbNav } from './BreadcrumbNav'

export function Layout() {
  return (
    <div className="relative min-h-screen flex flex-col">
      <Header />
      <div className="flex-1">
        <div className="container py-6">
          <BreadcrumbNav />
          <main className="mt-6">
            <Outlet />
          </main>
        </div>
      </div>
      <footer className="border-t py-6 md:py-0">
        <div className="container flex flex-col items-center justify-between gap-4 md:h-14 md:flex-row">
          <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
            Built with Vite, React, and Tailwind CSS. © {new Date().getFullYear()} optimex developers.
          </p>
        </div>
      </footer>
    </div>
  )
}
