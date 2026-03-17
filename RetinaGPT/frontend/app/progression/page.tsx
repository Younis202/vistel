'use client'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import Shell from '@/components/Shell'
import { analyzeProgression } from '@/lib/api'
import type { ProgressionReport } from '@/types'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

interface Visit { date: string; file: File | null; preview: string | null }

export default function ProgressionPage() {
  const [pid, setPid]       = useState('')
  const [visits, setVisits] = useState<Visit[]>([
    { date: '', file: null, preview: null },
    { date: '', file: null, preview: null },
  ])
  const [loading, setLoading] = useState(false)
  const [report,  setReport]  = useState<ProgressionReport | null>(null)

  const addVisit = () => setVisits(v => [...v, { date: '', file: null, preview: null }])
  const removeVisit = (i: number) => setVisits(v => v.filter((_, j) => j !== i))

  const setVisitFile = (i: number, f: File) => setVisits(v =>
    v.map((vi, j) => j === i ? { ...vi, file: f, preview: URL.createObjectURL(f) } : vi)
  )

  const run = async () => {
    if (!pid) return toast.error('Enter a patient ID')
    const valid = visits.filter(v => v.file && v.date)
    if (valid.length < 2) return toast.error('Need at least 2 visits with date and image')
    setLoading(true)
    try {
      const data = await Promise.all(valid.map(async v => {
        const b64 = await new Promise<string>((res, rej) => {
          const reader = new FileReader()
          reader.onload = () => res((reader.result as string).split(',')[1])
          reader.onerror = rej
          reader.readAsDataURL(v.file!)
        })
        return { visit_date: v.date, image_b64: b64 }
      }))
      const r = await analyzeProgression(pid, data)
      setReport(r)
      toast.success('Analysis complete')
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Analysis failed')
    } finally { setLoading(false) }
  }

  const chartData = report?.dr_grades.map((g, i) => ({
    label: report.visit_dates[i]?.slice(0, 10) || `V${i+1}`, grade: g,
  })) || []

  const trendColor = report
    ? { worsening: 'var(--red)', stable: 'var(--forest2)', improving: 'var(--forest2)' }[report.overall_trend] || 'var(--ink3)'
    : 'var(--ink3)'

  const riskColor = report
    ? { low: 'var(--forest2)', moderate: 'var(--amber)', high: 'var(--red)', critical: '#6B1818' }[report.risk_level] || 'var(--ink3)'
    : 'var(--ink3)'

  return (
    <Shell>
      <div className="topbar">
        <div className="topbar-left"><h1>Patient history</h1><p>Longitudinal progression analysis</p></div>
      </div>
      <div className="content">
        <div className="g2c">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="card">
              <div className="card-head"><span className="card-title">Patient</span></div>
              <div className="card-body">
                <div className="label" style={{ marginBottom: 6 }}>Patient ID</div>
                <input className="field" value={pid} onChange={e => setPid(e.target.value)} placeholder="e.g. P-00124" />
              </div>
            </div>

            {visits.map((v, i) => (
              <motion.div key={i} className="card" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
                <div className="card-head">
                  <span className="card-title" style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>Visit {i + 1}</span>
                  {visits.length > 2 && (
                    <button style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--ink4)', fontSize: 18, lineHeight: 1, padding: 0 }}
                      onClick={() => removeVisit(i)}>×</button>
                  )}
                </div>
                <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <input type="date" className="field" value={v.date} onChange={e => setVisits(vs => vs.map((vi, j) => j === i ? { ...vi, date: e.target.value } : vi))} />
                  <label style={{ cursor: 'pointer' }}>
                    <div style={{
                      border: `1.5px dashed ${v.preview ? 'var(--forest3)' : 'var(--paper3)'}`,
                      borderRadius: 7, padding: v.preview ? 0 : '16px 12px', textAlign: 'center',
                      background: v.preview ? 'transparent' : 'var(--paper2)', height: v.preview ? 80 : undefined,
                      overflow: 'hidden', cursor: 'pointer',
                    }}>
                      {v.preview
                        ? <img src={v.preview} style={{ width: '100%', height: 80, objectFit: 'cover', borderRadius: 6 }} alt="" />
                        : <p style={{ fontSize: 12, color: 'var(--ink3)' }}>Upload fundus image</p>
                      }
                    </div>
                    <input type="file" accept="image/*" style={{ display: 'none' }} onChange={e => e.target.files?.[0] && setVisitFile(i, e.target.files[0])} />
                  </label>
                </div>
              </motion.div>
            ))}

            <button style={{ width: '100%', padding: '10px', border: '1.5px dashed var(--paper3)', borderRadius: 8, background: 'none', color: 'var(--ink3)', fontSize: 13, cursor: 'pointer', fontFamily: 'var(--sans)' }} onClick={addVisit}>
              + Add visit
            </button>

            <button className="btn btn-forest btn-lg" onClick={run} disabled={loading}>
              {loading ? 'Analyzing…' : 'Analyze progression'}
            </button>
          </div>

          <AnimatePresence>
            {report ? (
              <motion.div initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div className="card">
                  <div className="card-body">
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                      <div>
                        <div className="label" style={{ marginBottom: 6 }}>Overall trend</div>
                        <div style={{ fontSize: 22, fontFamily: 'var(--serif)', color: trendColor, letterSpacing: '-0.02em', textTransform: 'capitalize' }}>
                          {report.overall_trend}
                        </div>
                      </div>
                      <div>
                        <div className="label" style={{ marginBottom: 6 }}>12-month risk</div>
                        <div style={{ fontSize: 22, fontFamily: 'var(--serif)', color: riskColor, letterSpacing: '-0.02em', textTransform: 'capitalize' }}>
                          {report.risk_level}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--ink3)', fontFamily: 'var(--mono)' }}>
                          {Math.round(report.risk_12m * 100)}% probability
                        </div>
                      </div>
                      <div>
                        <div className="label" style={{ marginBottom: 4 }}>Visits</div>
                        <div style={{ fontSize: 24, fontFamily: 'var(--serif)', color: 'var(--ink)' }}>{report.num_visits}</div>
                      </div>
                      <div>
                        <div className="label" style={{ marginBottom: 4 }}>Grade change</div>
                        <div style={{ fontSize: 24, fontFamily: 'var(--serif)', color: report.grade_change > 0 ? 'var(--red)' : report.grade_change < 0 ? 'var(--forest2)' : 'var(--ink3)' }}>
                          {report.grade_change > 0 ? '+' : ''}{report.grade_change}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {chartData.length > 0 && (
                  <div className="card">
                    <div className="card-head"><span className="card-title">DR grade over time</span></div>
                    <div style={{ padding: '16px 20px 20px' }}>
                      <ResponsiveContainer width="100%" height={160}>
                        <LineChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                          <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#7A7A72', fontFamily: 'Geist Mono' }} axisLine={false} tickLine={false} />
                          <YAxis domain={[0,4]} ticks={[0,1,2,3,4]} tick={{ fontSize: 10, fill: '#7A7A72', fontFamily: 'Geist Mono' }} axisLine={false} tickLine={false} />
                          <ReferenceLine y={2} stroke="var(--amber)" strokeDasharray="3 3" strokeOpacity={0.5} />
                          <Tooltip contentStyle={{ background: '#FAFAF7', border: '1px solid #E8E8E0', borderRadius: 7, fontSize: 12, fontFamily: 'Geist' }} />
                          <Line type="monotone" dataKey="grade" stroke="var(--forest3)" strokeWidth={2}
                            dot={{ fill: 'var(--forest3)', r: 4, strokeWidth: 2, stroke: '#FAFAF7' }}
                            activeDot={{ r: 6, fill: 'var(--forest)' }} />
                        </LineChart>
                      </ResponsiveContainer>
                      <p style={{ fontSize: 10, color: 'var(--ink3)', fontFamily: 'var(--mono)', marginTop: 6 }}>
                        Dashed line = referral threshold
                      </p>
                    </div>
                  </div>
                )}

                {report.new_lesions.length > 0 && (
                  <div style={{ borderLeft: '2px solid var(--amber)', background: 'var(--amber-pale)', borderRadius: '0 8px 8px 0', padding: '13px 16px' }}>
                    <div style={{ fontSize: 10, fontFamily: 'var(--mono)', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--amber2)', fontWeight: 500, marginBottom: 8 }}>
                      New lesions since baseline
                    </div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {report.new_lesions.map(l => (
                        <span key={l} style={{ fontSize: 12, padding: '3px 9px', background: 'rgba(184,134,11,0.12)', color: 'var(--amber2)', borderRadius: 4, border: '1px solid rgba(184,134,11,0.2)' }}>
                          {l}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="rec-box">
                  <div className="rec-label">Recommendation</div>
                  <div className="rec-text">{report.recommendation}</div>
                </div>
              </motion.div>
            ) : (
              <div className="card" style={{ minHeight: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <p style={{ fontSize: 13, color: 'var(--ink3)' }}>Add visits and analyze to see progression</p>
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </Shell>
  )
}
