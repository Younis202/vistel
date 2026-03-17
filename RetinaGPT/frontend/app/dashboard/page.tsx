'use client'
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import Shell from '@/components/Shell'
import GradeBadge from '@/components/GradeBadge'
import { getCaseStats, getCases } from '@/lib/api'
import type { CaseStats, CaseEntry } from '@/types'
import { formatDistanceToNow } from 'date-fns'
import Link from 'next/link'

const GRADE_COLORS = ['#1A3A2A','#3D7A58','#B8860B','#C0341D','#8B2020']
const GRADE_NAMES  = ['No DR','Mild','Moderate','Severe','Proliferative']

const EyeIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
    <circle cx="12" cy="12" r="3"/>
  </svg>
)

const fade = { initial: { opacity: 0, y: 12 }, animate: { opacity: 1, y: 0 } }

export default function DashboardPage() {
  const [stats, setStats] = useState<CaseStats | null>(null)
  const [cases, setCases] = useState<CaseEntry[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getCaseStats(), getCases({ limit: 6 })])
      .then(([s, c]) => { setStats(s); setCases(c.cases) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const chartData = GRADE_NAMES.map((name, i) => ({
    name,
    value: stats?.dr_grade_distribution?.[String(i)] || 0,
    color: GRADE_COLORS[i],
  }))

  const referPct = stats
    ? Math.round((stats.referable_cases / Math.max(stats.total_cases, 1)) * 100)
    : 0

  const STATS = [
    { label: 'Total scans',    value: stats?.total_cases  ?? 0, note: 'All time' },
    { label: 'This week',      value: stats?.this_week    ?? 0, note: '' },
    { label: 'Referable',      value: `${referPct}%`,           note: `${stats?.referable_cases ?? 0} cases` },
    { label: 'Today',          value: stats?.today        ?? 0, note: '' },
  ]

  return (
    <Shell>
      <div className="topbar">
        <div className="topbar-left">
          <h1>Dashboard</h1>
          <p>Clinical AI overview</p>
        </div>
        <div className="topbar-right">
          <Link href="/analyze">
            <button className="btn btn-forest">New analysis</button>
          </Link>
        </div>
      </div>

      <div className="content">
        {/* Stats */}
        <div className="stat-grid" style={{ marginBottom: 24 }}>
          {STATS.map((s, i) => (
            <motion.div key={i} className="stat-card" {...fade} transition={{ delay: i * 0.05 }}>
              <div className="stat-label">{s.label}</div>
              <div className="stat-value">{loading ? '—' : s.value}</div>
              {s.note && <div className="stat-note">{s.note}</div>}
            </motion.div>
          ))}
        </div>

        <div className="g2c">
          {/* Chart */}
          <motion.div className="card" {...fade} transition={{ delay: 0.1 }}>
            <div className="card-head">
              <span className="card-title">DR distribution</span>
              <span className="card-meta">{stats?.total_cases ?? 0} scans</span>
            </div>
            <div style={{ padding: '20px 20px 16px' }}>
              {!loading && chartData.some(d => d.value > 0) ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={chartData} barSize={24} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#7A7A72', fontFamily: 'Geist Mono' }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 10, fill: '#7A7A72', fontFamily: 'Geist Mono' }} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{ background: '#FAFAF7', border: '1px solid #E8E8E0', borderRadius: 7, fontSize: 12, fontFamily: 'Geist' }}
                      cursor={{ fill: 'rgba(0,0,0,0.03)' }}
                    />
                    <Bar dataKey="value" radius={[3,3,0,0]}>
                      {chartData.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.9} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <span style={{ fontSize: 13, color: 'var(--ink3)' }}>No data yet</span>
                </div>
              )}
            </div>
          </motion.div>

          {/* Recent cases */}
          <motion.div className="card" {...fade} transition={{ delay: 0.15 }}>
            <div className="card-head">
              <span className="card-title">Recent cases</span>
              <Link href="/reports" style={{ fontSize: 12, color: 'var(--forest3)', textDecoration: 'none' }}>
                View all →
              </Link>
            </div>

            {loading ? (
              <div style={{ padding: '20px' }}>
                {[...Array(4)].map((_, i) => (
                  <div key={i} style={{ height: 44, background: 'var(--paper2)', borderRadius: 6, marginBottom: 8 }} />
                ))}
              </div>
            ) : cases.length === 0 ? (
              <div style={{ padding: '48px 20px', textAlign: 'center' }}>
                <div style={{ marginBottom: 12, color: 'var(--ink4)' }}><EyeIcon /></div>
                <p style={{ fontSize: 13, color: 'var(--ink3)' }}>No cases yet</p>
                <Link href="/analyze" style={{ fontSize: 12, color: 'var(--forest3)', textDecoration: 'none', marginTop: 8, display: 'block' }}>
                  Analyze your first image →
                </Link>
              </div>
            ) : (
              cases.map((c, i) => (
                <motion.div
                  key={c.id}
                  className="case-row"
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.15 + i * 0.04 }}
                >
                  <div className="case-eye-icon" style={{ color: 'var(--ink3)' }}><EyeIcon /></div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, color: 'var(--ink)', fontFamily: 'var(--mono)', fontWeight: 500 }}>
                      {c.patient_id !== 'Unknown' ? c.patient_id : c.id.slice(0, 8)}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--ink3)', marginTop: 1 }}>
                      {formatDistanceToNow(new Date(c.created_at), { addSuffix: true })}
                    </div>
                  </div>
                  <GradeBadge grade={c.dr_grade} showRefer />
                  <span style={{ fontSize: 11, color: 'var(--ink3)', fontFamily: 'var(--mono)', width: 32, textAlign: 'right' }}>
                    {Math.round(c.dr_confidence * 100)}%
                  </span>
                </motion.div>
              ))
            )}
          </motion.div>
        </div>
      </div>
    </Shell>
  )
}
