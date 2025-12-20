import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Home } from './pages/Home'
import { Installation } from './pages/Installation'
import { Theory } from './pages/Theory'
import { Examples } from './pages/Examples'
import { ApiReference } from './pages/ApiReference'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="installation" element={<Installation />} />
          <Route path="theory" element={<Theory />} />
          <Route path="examples" element={<Examples />} />
          <Route path="api" element={<ApiReference />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
