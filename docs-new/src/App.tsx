import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from './components/ThemeProvider'
import { Layout } from './components/Layout'
import { Home } from './pages/Home'
import { Installation } from './pages/Installation'
import { Theory } from './pages/Theory'
import { Examples } from './pages/Examples'
import { ApiOverview } from './pages/ApiOverview'
import { ApiModule } from './pages/ApiModule'

function App() {
  return (
    <ThemeProvider defaultTheme="system" storageKey="optimex-ui-theme">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="installation" element={<Installation />} />
            <Route path="theory" element={<Theory />} />
            <Route path="examples" element={<Examples />} />
            <Route path="api" element={<ApiOverview />} />
            <Route path="api/:module" element={<ApiModule />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
