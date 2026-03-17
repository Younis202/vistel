'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import { checkHealth } from '@/lib/api'

const NAV = [
  { section: 'Overview', items: [{ href: '/dashboard', label: 'Dashboard' }] },
  { section: 'Clinical', items: [
    { href: '/analyze',     label: 'New analysis' },
    { href: '/search',      label: 'Similar cases' },
    { href: '/progression', label: 'Patient history' },
  ]},
  { section: 'Records', items: [
    { href: '/reports', label: 'All reports' },
  ]},
]

export default function Sidebar() {
  const path = usePathname()
  const [online, setOnline] = useState<boolean | null>(null)

  useEffect(() => {
    checkHealth().then(setOnline)
    const id = setInterval(() => checkHealth().then(setOnline), 30000)
    return () => clearInterval(id)
  }, [])

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="wordmark">Retina<em>GPT</em></div>
        <div className="wordmark-sub">Ophthalmology AI</div>
      </div>

      <nav className="sidebar-nav">
        {NAV.map(({ section, items }) => (
          <div key={section} className="nav-section">
            <div className="nav-section-label">{section}</div>
            {items.map(({ href, label }) => {
              const active = path === href || (href !== '/dashboard' && path.startsWith(href))
              return (
                <Link key={href} href={href} className={`nav-item${active ? ' active' : ''}`}>
                  <span className="nav-dot" />
                  {label}
                </Link>
              )
            })}
          </div>
        ))}

        <Link href="/analyze" className="nav-cta">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>
          </svg>
          New analysis
        </Link>
      </nav>

      <div className="sidebar-foot">
        <div className="api-pill">
          <div className={`api-dot${online === false ? ' offline' : ''}`} />
          <span className="api-text">
            {online === null ? 'Connecting...' : online ? 'API connected' : 'API offline'}
          </span>
        </div>
      </div>
    </aside>
  )
}
