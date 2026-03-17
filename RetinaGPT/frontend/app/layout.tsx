import type { Metadata } from 'next'
import { Toaster } from 'react-hot-toast'
import './globals.css'

export const metadata: Metadata = {
  title: 'RetinaGPT',
  description: 'AI Ophthalmology Platform',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: '#FAFAF7',
              color: '#0E0E0C',
              border: '1px solid #E8E8E0',
              fontSize: '13px',
              fontFamily: "'Geist', system-ui, sans-serif",
              borderRadius: '8px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
            },
            success: { iconTheme: { primary: '#2C5C42', secondary: '#EBF3EE' } },
            error:   { iconTheme: { primary: '#8B2020', secondary: '#FBF0EE' } },
          }}
        />
        {children}
      </body>
    </html>
  )
}
