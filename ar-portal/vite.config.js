import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Local dev: proxy every backend path to the FastAPI dev server so the
// frontend can use same-origin relative URLs (see src/api.js). A deployed
// build talks to VITE_API_URL directly and never uses this.
const API = 'http://127.0.0.1:8000'
const proxied = [
  '/extract',
  '/results',
  '/download',
  '/health',
  '/template-sheets',
  '/process-vendor',
  '/onboarding',
  '/auth',
  '/vendors',
  '/customers',
  '/business-central',
]

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    proxy: Object.fromEntries(
      proxied.map(path => [path, { target: API, changeOrigin: true }])
    ),
  },
})
