'use client'
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'
import Shell from '@/components/Shell'
import GradeBadge from '@/components/GradeBadge'
import { getCases, deleteCase } from '@/lib/api'
import type { CaseEntry } from '@/types'
import { format } from 'date-fns'

export default function ReportsPage() {
  const [cases, setCases]   = useState<CaseEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [gradeF, setGradeF] = useState('')
  const [referF, setReferF] = useState(false)

  const load = () => {
    setLoading(true)
    getCases({ limit: 100, patient_id: search || undefined, dr_grade: gradeF !== '' ? +gradeF : undefined, refer_only: referF || undefined })
      .then(r => setCases(r.cases))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [search, gradeF, referF])

  const del = async (id: string) => {
    try { await deleteCase(id); setCases(c => c.filter(x => x.id !== id)); toast.success('Deleted') }
    catch { toast.error('Could not delete') }
  }

  return (
    <Shell>
      <div className="topbar">
        <div className="topbar-left"><h1>All reports</h1><p>{cases.length} cases</p></div>
        <div className="topbar-right">
          <button className="btn btn-ghost" onClick={load}>Refresh</button>
        </div>
      </div>
      <div className="content">
        {/* Filters */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <svg style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--ink3)' }} width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <input className="field" value={search} onChange={e => setSearch(e.target.value)} placeholder="Search patient ID…" style={{ paddingLeft: 32 }} />
          </div>
          <select className="field" style={{ width: 160 }} value={gradeF} onChange={e => setGradeF(e.target.value)}>
            <option value="">All grades</option>
            {[0,1,2,3,4].map(g => <option key={g} value={g}>Grade {g}</option>)}
          </select>
          <button className="btn" style={{ border: `1px solid ${referF ? 'var(--amber)' : 'var(--paper3)'}`, background: referF ? 'var(--amber-pale)' : 'transparent', color: referF ? 'var(--amber2)' : 'var(--ink2)' }}
            onClick={() => setReferF(r => !r)}>
            Referable only
          </button>
        </div>

        <div className="card">
          <table className="data-table">
            <thead>
              <tr>
                <th>Patient</th><th>Date</th><th>DR grade</th><th>Confidence</th><th>Quality</th><th>Status</th><th></th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                [...Array(5)].map((_, i) => (
                  <tr key={i}>
                    {[...Array(7)].map((_, j) => (
                      <td key={j}><div style={{ height: 14, background: 'var(--paper2)', borderRadius: 3, width: '70%' }} /></td>
                    ))}
                  </tr>
                ))
              ) : cases.length === 0 ? (
                <tr><td colSpan={7} style={{ textAlign: 'center', padding: '48px 20px', color: 'var(--ink3)' }}>No cases found</td></tr>
              ) : cases.map((c, i) => (
                <motion.tr key={c.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.02 }}>
                  <td>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--ink)', fontWeight: 500 }}>
                      {c.patient_id !== 'Unknown' ? c.patient_id : '—'}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--ink3)', fontFamily: 'var(--mono)', marginTop: 1 }}>{c.id.slice(0, 8)}</div>
                  </td>
                  <td style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink3)' }}>
                    {format(new Date(c.created_at), 'MMM d, yyyy HH:mm')}
                  </td>
                  <td><GradeBadge grade={c.dr_grade} showRefer /></td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 60, height: 3, background: 'var(--paper3)', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${c.dr_confidence * 100}%`, background: 'var(--forest3)', borderRadius: 2 }} />
                      </div>
                      <span style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--ink3)' }}>
                        {Math.round(c.dr_confidence * 100)}%
                      </span>
                    </div>
                  </td>
                  <td>
                    <span style={{ fontSize: 11, padding: '3px 8px', borderRadius: 4, fontFamily: 'var(--mono)',
                      background: c.quality_adequate ? 'rgba(44,92,66,0.08)' : 'rgba(139,32,32,0.08)',
                      color: c.quality_adequate ? 'var(--forest2)' : 'var(--red)' }}>
                      {c.quality_adequate ? 'Good' : 'Poor'}
                    </span>
                  </td>
                  <td>
                    <span style={{ fontSize: 11, padding: '3px 8px', borderRadius: 4, fontFamily: 'var(--mono)',
                      background: c.dr_refer ? 'var(--amber-pale)' : 'var(--paper2)',
                      color: c.dr_refer ? 'var(--amber2)' : 'var(--ink3)',
                      border: `1px solid ${c.dr_refer ? 'rgba(184,134,11,0.2)' : 'var(--paper3)'}` }}>
                      {c.dr_refer ? 'REFER' : 'Routine'}
                    </span>
                  </td>
                  <td>
                    <button onClick={() => del(c.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--ink4)', padding: 4, borderRadius: 4, transition: 'color 0.12s' }}
                      onMouseEnter={e => (e.currentTarget.style.color = 'var(--red)')}
                      onMouseLeave={e => (e.currentTarget.style.color = 'var(--ink4)')}>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
                    </button>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Shell>
  )
}
