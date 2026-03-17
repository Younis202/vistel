'use client'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import Shell from '@/components/Shell'
import GradeBadge from '@/components/GradeBadge'
import { searchSimilar } from '@/lib/api'
import type { SearchResult } from '@/types'

export default function SearchPage() {
  const [file, setFile]     = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [k, setK]           = useState(10)
  const [grade, setGrade]   = useState<number | undefined>()
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<SearchResult[] | null>(null)
  const [meta, setMeta]     = useState<{ size: number; ms: number } | null>(null)

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: f => { if (f[0]) { setFile(f[0]); setPreview(URL.createObjectURL(f[0])); setResults(null) } },
    accept: { 'image/*': [] }, maxFiles: 1,
  })

  const run = async () => {
    if (!file) return toast.error('Select a query image')
    setLoading(true)
    try {
      const r = await searchSimilar(file, k, grade)
      setResults(r.results)
      setMeta({ size: r.index_size, ms: r.search_time_ms })
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Search failed — build the index first')
    } finally { setLoading(false) }
  }

  return (
    <Shell>
      <div className="topbar">
        <div className="topbar-left"><h1>Similar cases</h1><p>FAISS semantic vector search</p></div>
      </div>
      <div className="content">
        <div className="g2c">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="card">
              <div className="card-head"><span className="card-title">Query image</span></div>
              <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div {...getRootProps()} className={`drop-zone${isDragActive ? ' drag' : ''}${preview ? ' has-file' : ''}`} style={{ height: 160, padding: preview ? 0 : undefined }}>
                  <input {...getInputProps()} />
                  {preview
                    ? <img src={preview} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 7 }} />
                    : <><div className="drop-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div><div className="drop-title">Drop query image</div></>
                  }
                </div>

                <div>
                  <div className="label" style={{ marginBottom: 6 }}>Results (k = {k})</div>
                  <input type="range" min={1} max={50} value={k} onChange={e => setK(+e.target.value)} style={{ width: '100%', accentColor: 'var(--forest3)' }} />
                </div>

                <div>
                  <div className="label" style={{ marginBottom: 6 }}>Filter by grade</div>
                  <select className="field" value={grade ?? ''} onChange={e => setGrade(e.target.value === '' ? undefined : +e.target.value)}>
                    <option value="">All grades</option>
                    {[0,1,2,3,4].map(g => <option key={g} value={g}>Grade {g}</option>)}
                  </select>
                </div>

                <button className="btn btn-forest btn-lg" onClick={run} disabled={!file || loading}>
                  {loading ? 'Searching…' : 'Find similar cases'}
                </button>

                {meta && <p style={{ fontSize: 11, color: 'var(--ink3)', textAlign: 'center', fontFamily: 'var(--mono)' }}>
                  {meta.size.toLocaleString()} embeddings · {meta.ms.toFixed(1)}ms
                </p>}
              </div>
            </div>
          </div>

          <div>
            <AnimatePresence>
              {results !== null && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {results.length === 0 ? (
                    <div className="card" style={{ padding: '48px 20px', textAlign: 'center' }}>
                      <p style={{ fontSize: 13, color: 'var(--ink3)' }}>No results. Build the index first.</p>
                    </div>
                  ) : results.map((r, i) => (
                    <motion.div key={r.image_id} className="search-card" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                      <div style={{ width: 32, height: 32, borderRadius: '50%', background: 'var(--paper3)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        <span style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--ink3)', fontWeight: 500 }}>{r.rank + 1}</span>
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 13, color: 'var(--ink)', fontFamily: 'var(--mono)' }}>{r.image_id}</div>
                        {r.dr_grade !== null && <GradeBadge grade={r.dr_grade} />}
                        {r.dataset && <div style={{ fontSize: 11, color: 'var(--ink3)', marginTop: 2 }}>{r.dataset}</div>}
                      </div>
                      <div style={{ textAlign: 'right', flexShrink: 0 }}>
                        <div className="score-num">{(r.score * 100).toFixed(1)}%</div>
                        <div style={{ fontSize: 10, color: 'var(--ink3)', fontFamily: 'var(--mono)', marginTop: 1 }}>similarity</div>
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              )}
              {results === null && (
                <div className="card" style={{ minHeight: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <p style={{ fontSize: 13, color: 'var(--ink3)' }}>Upload an image to search</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </Shell>
  )
}
