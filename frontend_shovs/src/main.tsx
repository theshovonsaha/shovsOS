import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { TrustReceiptsApp } from './components/TrustReceiptsApp.tsx'

const rootElement = window.location.pathname === '/trust-receipts'
  ? <TrustReceiptsApp />
  : <App />

createRoot(document.getElementById('root')!).render(
  rootElement
)
