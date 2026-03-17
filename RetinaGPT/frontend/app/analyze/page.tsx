'use client'
import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import Shell from '@/components/Shell'
import GradeBadge from '@/components/GradeBadge'
import { analyzeImage, downloadPDF, b64 } from '@/lib/api'
import type { AnalysisResult } from '@/types'

const UploadIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
)

const EyeIcon = ({ size = 40 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
  </svg>
)

const LESION_NAMES: Record<string, string> = {
  microaneurysm: 'Microaneurysm', hemorrhage: 'Hemorrhage',
  hard_exudate: 'Hard exudate', soft_exudate: 'Cotton-wool spot',
  neovascularization: 'Neovascularization', drusen: 'Drusen',
}

type Tab = 'original' | 'gradcam' | 'attention' | 'vessel'

export default function AnalyzePage() {
  const [file,    setFile]    = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [pid,     setPid]     = useState('')
  const [explain, setExplain] = useState(true)
  const [segment, setSegment] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState<AnalysisResult | null>(null)
  const [tab,     setTab]     = useState<Tab>('original')
  const [pdfBusy, setPdfBusy] = useState(false)

  const onDrop = useCallback((files: File[]) => {
    if (!files[0]) return
    setFile(files[0])
    setPreview(URL.createObjectURL(files[0]))
    setResult(null)
    setTab('original')
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/*': ['.png','.jpg','.jpeg','.tif','.tiff'] }, maxFiles: 1, disabled: loading,
  })

  const run = async () => {
    if (!file) return toast.error('Select a fundus image first')
    setLoading(true)
    try {
      const r = await analyzeImage(file, { explain, segment, patientId: pid || undefined })
      setResult(r)
      if (r.explainability.gradcam_image) setTab('gradcam')
      toast.success('Analysis complete')
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Analysis failed — is the API running?')
    } finally {
      setLoading(false)
    }
  }

  const handlePDF = async () => {
    if (!file || !result) return
    setPdfBusy(true)
    try {
      const blob = await downloadPDF(file, pid || result.image_id)
      const a = Object.assign(document.createElement('a'), {
        href: URL.createObjectURL(blob),
        download: `retina_report_${result.image_id}.pdf`,
      })
      a.click()
      toast.success('Report downloaded')
    } catch { toast.error('PDF generation failed') }
    finally { setPdfBusy(false) }
  }

  const imgSrc = () => {
    if (!result) return preview
    if (tab === 'gradcam')  return b64(result.explainability.gradcam_image)  ?? preview
    if (tab === 'attention')return b64(result.explainability.attention_image) ?? preview
    if (tab === 'vessel')   return b64(result.segmentation.vessel_mask)      ?? preview
    return preview
  }

  const TABS: { key: Tab; label: string; avail: boolean }[] = [
    { key: 'original',  label: 'Original',  avail: true },
    { key: 'gradcam',   label: 'Grad-CAM',  avail: !!result?.explainability.gradcam_image },
    { key: 'attention', label: 'Attention', avail: !!result?.explainability.attention_image },
    { key: 'vessel',    label: 'Vessels',   avail: !!result?.segmentation.vessel_mask },
  ]

  return (
    <Shell>
      <div className="topbar">
        <div className="topbar-left">
          <h1>New analysis</h1>
          <p>Upload a retinal fundus image</p>
        </div>
        <div className="topbar-right">
          {result && (
            <button className="btn btn-ghost" onClick={handlePDF} disabled={pdfBusy}>
              {pdfBusy ? 'Generating...' : 'Download PDF'}
            </button>
          )}
        </div>
      </div>

      <div className="content">
        <div className="g2c">
          {/* Upload panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="card">
              <div className="card-head">
                <span className="card-title">Fundus image</span>
                <span className="card-meta">PNG · JPG · TIFF</span>
              </div>
              <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

                {/* Dropzone */}
                <div
                  {...getRootProps()}
                  className={`drop-zone${isDragActive ? ' drag' : ''}${preview ? ' has-file' : ''}`}
                  style={{ padding: preview ? 0 : undefined, height: preview ? 200 : undefined }}
                >
                  <input {...getInputProps()} />
                  {preview ? (
                    <div style={{ position: 'relative', height: '100%' }}>
                      <img src={preview} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 7 }} />
                      <div style={{
                        position: 'absolute', inset: 0, background: 'rgba(26,58,42,0)', display: 'flex',
                        alignItems: 'center', justifyContent: 'center', borderRadius: 7, transition: 'background 0.15s',
                      }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(26,58,42,0.5)')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'rgba(26,58,42,0)')}
                      >
                        <span style={{ color: 'white', fontSize: 13, opacity: 0, transition: 'opacity 0.15s' }}
                          onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
                          onMouseLeave={e => (e.currentTarget.style.opacity = '0')}
                        >Click to change</span>
                      </div>
                      <div style={{
                        position: 'absolute', bottom: 10, left: 10,
                        background: 'rgba(10,14,12,0.7)', borderRadius: 4,
                        padding: '3px 8px', fontSize: 11, color: 'rgba(250,250,247,0.8)',
                        fontFamily: 'var(--mono)',
                      }}>
                        {file?.name}
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="drop-icon" style={{ color: 'var(--ink3)' }}><UploadIcon /></div>
                      <div className="drop-title">{isDragActive ? 'Release to upload' : 'Drop fundus image here'}</div>
                      <div className="drop-sub">or click to browse</div>
                    </>
                  )}
                </div>

                {/* Patient ID */}
                <div>
                  <div className="label" style={{ marginBottom: 6 }}>Patient ID</div>
                  <input className="field" value={pid} onChange={e => setPid(e.target.value)} placeholder="e.g. P-00124" />
                </div>

                {/* Toggles */}
                <div style={{ borderTop: '1px solid var(--paper3)', paddingTop: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {[
                    { id: 'explain', label: 'Grad-CAM explanation', hint: 'Highlight diagnostic regions', val: explain, set: () => setExplain(v => !v) },
                    { id: 'segment', label: 'Vessel segmentation',  hint: 'Extract vascular structure',  val: segment, set: () => setSegment(v => !v) },
                  ].map(t => (
                    <div key={t.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div>
                        <div style={{ fontSize: 13, color: 'var(--ink2)' }}>{t.label}</div>
                        <div style={{ fontSize: 11, color: 'var(--ink3)', marginTop: 1 }}>{t.hint}</div>
                      </div>
                      <button className={`toggle-track${t.val ? '' : ' off'}`} onClick={t.set}>
                        <div className="toggle-thumb" />
                      </button>
                    </div>
                  ))}
                </div>

                <motion.button
                  className="btn btn-forest btn-lg"
                  onClick={run}
                  disabled={!file || loading}
                  whileHover={!loading && file ? { scale: 1.01 } : {}}
                  whileTap={!loading && file ? { scale: 0.99 } : {}}
                >
                  {loading ? (
                    <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <motion.svg animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                        width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
                      </motion.svg>
                      Analyzing…
                    </span>
                  ) : 'Analyze image'}
                </motion.button>
              </div>
            </div>
          </div>

          {/* Results panel */}
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                style={{ display: 'flex', flexDirection: 'column', gap: 14 }}
              >
                {/* Image viewer */}
                <div className="card">
                  <div className="card-body" style={{ paddingBottom: 14 }}>
                    <div className="img-viewer" style={{ height: 200 }}>
                      <AnimatePresence mode="wait">
                        <motion.img
                          key={tab}
                          src={imgSrc() || ''}
                          alt=""
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.18 }}
                          style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 8 }}
                        />
                      </AnimatePresence>
                      <div style={{
                        position: 'absolute', bottom: 10, right: 10,
                        background: 'rgba(10,14,12,0.65)', borderRadius: 4,
                        padding: '3px 8px', fontSize: 10, color: 'rgba(250,250,247,0.7)', fontFamily: 'var(--mono)',
                      }}>
                        {result.inference_time_ms.toFixed(0)}ms
                      </div>
                    </div>
                    <div className="img-tabs">
                      {TABS.map(t => (
                        <button
                          key={t.key}
                          className={`img-tab${tab === t.key ? ' active' : ''}`}
                          disabled={!t.avail}
                          onClick={() => t.avail && setTab(t.key)}
                          style={!t.avail ? { opacity: 0.35, cursor: 'not-allowed' } : {}}
                        >{t.label}</button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Grade hero */}
                <div className="card">
                  <div className="card-body">
                    <div className="grade-hero">
                      <div>
                        <div className="grade-number">{result.dr_grading.grade}</div>
                        <div style={{ fontSize: 14, color: 'var(--forest2)', fontWeight: 500, marginTop: 4 }}>
                          {result.dr_grading.label}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--forest3)', marginTop: 2, fontFamily: 'var(--mono)' }}>
                          Confidence {Math.round(result.dr_grading.confidence * 100)}%
                        </div>
                      </div>
                      {result.dr_grading.refer
                        ? <span className="refer-tag">Refer</span>
                        : <span className="ok-tag">Routine</span>
                      }
                    </div>

                    {/* Prob bars */}
                    <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
                      {['No DR','Mild','Moderate','Severe','Proliferative'].map((name, i) => {
                        const pct = (result.dr_grading.probabilities[i] || 0) * 100
                        return (
                          <div key={i} className="prob-row">
                            <div className="prob-name">{name}</div>
                            <div className="prob-track">
                              <motion.div
                                className="prob-fill"
                                initial={{ width: 0 }}
                                animate={{ width: `${pct}%` }}
                                transition={{ delay: i * 0.08, duration: 0.45 }}
                                style={{ background: ['#1A3A2A','#3D7A58','#B8860B','#C0341D','#8B2020'][i] }}
                              />
                            </div>
                            <div className="prob-pct">{pct.toFixed(1)}%</div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>

                {/* Mini cards */}
                <div className="card">
                  <div className="card-body" style={{ paddingBottom: 14 }}>
                    <div className="mini-grid">
                      <div className="mini-card">
                        <div className="mini-label">AMD</div>
                        <div className="mini-val" style={{ color: result.amd.stage > 0 ? 'var(--amber)' : 'var(--forest2)' }}>
                          {result.amd.label}
                        </div>
                        <div className="mini-sub">{Math.round(result.amd.confidence * 100)}% confidence</div>
                      </div>
                      <div className="mini-card">
                        <div className="mini-label">Glaucoma</div>
                        <div className="mini-val" style={{ color: result.glaucoma.suspect ? 'var(--amber)' : 'var(--forest2)' }}>
                          {result.glaucoma.suspect ? 'Suspect' : 'No suspicion'}
                        </div>
                        <div className="mini-sub">CDR {result.glaucoma.cup_disc_ratio.toFixed(2)}</div>
                      </div>
                      <div className="mini-card">
                        <div className="mini-label">Quality</div>
                        <div className="mini-val" style={{ color: result.quality.adequate ? 'var(--forest2)' : 'var(--red)' }}>
                          {result.quality.adequate ? 'Adequate' : 'Poor quality'}
                        </div>
                        <div className="mini-sub">{Math.round(result.quality.score * 100)}% score</div>
                      </div>
                      <div className="mini-card">
                        <div className="mini-label">Lesions</div>
                        <div className="mini-val" style={{
                          color: Object.values(result.lesions).some(l => l.present) ? 'var(--amber)' : 'var(--forest2)'
                        }}>
                          {Object.values(result.lesions).filter(l => l.present).length} detected
                        </div>
                        <div className="mini-sub">
                          {Object.entries(result.lesions).filter(([,v]) => v.present).map(([k]) =>
                            LESION_NAMES[k]?.split(' ')[0] || k
                          ).join(', ') || 'None found'}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recommendation */}
                {result.report.recommendation && (
                  <div style={{ padding: '0 0 4px' }}>
                    <div className="rec-box">
                      <div className="rec-label">Recommendation</div>
                      <div className="rec-text">{result.report.recommendation}</div>
                    </div>
                  </div>
                )}
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="card"
                style={{ height: '100%', minHeight: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                <div style={{ textAlign: 'center', color: 'var(--ink4)' }}>
                  <EyeIcon />
                  <p style={{ fontSize: 13, color: 'var(--ink3)', marginTop: 12 }}>
                    Results will appear here
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </Shell>
  )
}
