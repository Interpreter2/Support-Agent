import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/tickets': 'http://127.0.0.1:8000',
      '/runs': 'http://127.0.0.1:8000',
      '/approvals': 'http://127.0.0.1:8000',
    }
  }
})
