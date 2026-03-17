'use client'
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const DR_LABELS: Record<number, string> = {
  0: 'No Diabetic Retinopathy', 1: 'Mild Non-Proliferative DR',
  2: 'Moderate Non-Proliferative DR', 3: 'Severe Non-Proliferative DR',
  4: 'Proliferative Diabetic Retinopathy',
}

const GRADE_COLORS: Record<number, { bg: string; color: string; border: string }> = {
  0: { bg: '#EBF3EE', color: '#1A3A2A', border: 'rgba(44,92,66,0.2)' },
  1: { bg: '#EBF3EE', color: '#2C5C42', border: 'rgba(61,122,88,0.2)' },
  2: { bg: '#FBF5E6', color: '#7A5A00', border: 'rgba(184,134,11,0.2)' },
  3: { bg: '#FBF0EE', color: '#8B2020', border: 'rgba(139,32,32,0.2)' },
  4: { bg: '#FBF0EE', color: '#6B1818', border: 'rgba(107,24,24,0.25)' },
}

export default function PassportPage({ params }: { params: { token: string } }) {
  const [data,    setData]    = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(false)

  useEffect(() => {
    axios.get(`${BASE}/passport/${params.token}`)
      .then(r => setData(r.data))
      .catch(() => setError(true))
      .finally(() => setLoading(false))
  }, [params.token])

  if (loading) return (
    <div style={{ minHeight: '100vh', background: '#FAFAF7', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontFamily: 'DM Serif Display, serif', fontSize: 24, color: '#1A3A2A', marginBottom: 8 }}>
          Retina<em>GPT</em>
        </div>
        <p style={{ fontSize: 13, color: '#7A7A72' }}>Loading your report…</p>
      </div>
    </div>
  )

  if (error || !data) return (
    <div style={{ minHeight: '100vh', background: '#FAFAF7', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ textAlign: 'center', maxWidth: 400, padding: '0 24px' }}>
        <div style={{ fontFamily: 'DM Serif Display, serif', fontSize: 24, color: '#1A3A2A', marginBottom: 16 }}>
          Retina<em>GPT</em>
        </div>
        <h2 style={{ fontSize: 20, fontFamily: 'DM Serif Display, serif', color: '#0E0E0C', marginBottom: 8 }}>
          Link not found
        </h2>
        <p style={{ fontSize: 13, color: '#7A7A72', lineHeight: 1.6 }}>
          This link has expired or is no longer valid. Please contact your doctor for a new link.
        </p>
      </div>
    </div>
  )

  const grade  = data.dr_grade ?? 0
  const colors = GRADE_COLORS[grade] || GRADE_COLORS[0]
  const scanDate = data.scan_date ? new Date(data.scan_date).toLocaleDateString('en-GB', { year: 'numeric', month: 'long', day: 'numeric' }) : ''

  return (
    <div style={{ minHeight: '100vh', background: '#FAFAF7', fontFamily: 'Geist, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ borderBottom: '1px solid #E8E8E0', padding: '18px 24px', background: '#FAFAF7' }}>
        <div style={{ maxWidth: 600, margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ fontFamily: 'DM Serif Display, serif', fontSize: 20, color: '#1A3A2A' }}>
            Retina<em>GPT</em>
          </div>
          <div style={{ fontSize: 11, color: '#7A7A72', fontFamily: 'Geist Mono, monospace' }}>
            Retinal Analysis Report
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 600, margin: '0 auto', padding: '32px 24px' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>

          {/* Patient info */}
          <div style={{ marginBottom: 24 }}>
            <h1 style={{ fontFamily: 'DM Serif Display, serif', fontSize: 28, color: '#0E0E0C', letterSpacing: '-0.02em', marginBottom: 4 }}>
              Your retinal scan
            </h1>
            <p style={{ fontSize: 13, color: '#7A7A72' }}>
              Patient {data.patient_id} · {scanDate}
              {data.views > 0 && ` · Viewed ${data.views} time${data.views > 1 ? 's' : ''}`}
            </p>
          </div>

          {/* Grade card */}
          <div style={{
            background: colors.bg, border: `1px solid ${colors.border}`,
            borderRadius: 12, padding: 24, marginBottom: 20,
          }}>
            <div style={{ fontSize: 11, fontFamily: 'Geist Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.10em', color: colors.color, opacity: 0.7, marginBottom: 10 }}>
              AI Assessment
            </div>
            <div style={{ fontFamily: 'DM Serif Display, serif', fontSize: 52, color: colors.color, lineHeight: 1, letterSpacing: '-0.03em' }}>
              {grade}
            </div>
            <div style={{ fontSize: 16, color: colors.color, fontWeight: 500, marginTop: 6 }}>
              {DR_LABELS[grade] || data.dr_label}
            </div>
            {data.dr_refer && (
              <div style={{
                display: 'inline-flex', marginTop: 12, padding: '5px 12px',
                background: '#FBF0EE', border: '1px solid rgba(139,32,32,0.2)',
                borderRadius: 5, fontSize: 11, color: '#8B2020',
                fontFamily: 'Geist Mono, monospace', fontWeight: 500, letterSpacing: '0.08em', textTransform: 'uppercase',
              }}>
                Ophthalmology review recommended
              </div>
            )}
          </div>

          {/* Grad-CAM if available */}
          {data.gradcam_image && (
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 11, fontFamily: 'Geist Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.10em', color: '#7A7A72', marginBottom: 10 }}>
                AI Analysis — Highlighted regions
              </div>
              <img
                src={`data:image/png;base64,${data.gradcam_image}`}
                alt="Retinal scan with AI analysis"
                style={{ width: '100%', borderRadius: 10, border: '1px solid #E8E8E0' }}
              />
              <p style={{ fontSize: 11, color: '#7A7A72', marginTop: 6, lineHeight: 1.5 }}>
                The highlighted areas show where the AI focused when making its assessment.
              </p>
            </div>
          )}

          {/* Quality */}
          <div style={{
            background: '#F3F3EE', borderRadius: 8, padding: 16, marginBottom: 20,
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{
              width: 32, height: 32, borderRadius: '50%',
              background: data.quality_adequate ? '#EBF3EE' : '#FBF0EE',
              display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
            }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                stroke={data.quality_adequate ? '#1A3A2A' : '#8B2020'} strokeWidth="2">
                {data.quality_adequate
                  ? <><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></>
                  : <><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></>
                }
              </svg>
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, color: '#0E0E0C' }}>
                Image quality: {data.quality_adequate ? 'Good' : 'Limited'}
              </div>
              <div style={{ fontSize: 11, color: '#7A7A72', marginTop: 1 }}>
                {data.quality_adequate
                  ? 'Your scan was clear and suitable for analysis.'
                  : 'Image quality was limited. A repeat scan may be needed for certainty.'}
              </div>
            </div>
          </div>

          {/* Recommendation */}
          {data.recommendation && (
            <div style={{
              borderLeft: '2px solid #B8860B', background: '#FBF5E6',
              borderRadius: '0 8px 8px 0', padding: '14px 18px', marginBottom: 24,
            }}>
              <div style={{ fontSize: 10, fontFamily: 'Geist Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.10em', color: '#96700A', fontWeight: 500, marginBottom: 6 }}>
                Next steps
              </div>
              <p style={{ fontSize: 13, color: '#3D3D38', lineHeight: 1.6 }}>{data.recommendation}</p>
            </div>
          )}

          {/* Disclaimer */}
          <div style={{ borderTop: '1px solid #E8E8E0', paddingTop: 16 }}>
            <p style={{ fontSize: 11, color: '#A8A89E', lineHeight: 1.6 }}>
              This report was generated by RetinaGPT AI and is intended to assist clinical decision-making.
              It does not replace the judgment of a qualified ophthalmologist. Please discuss these results with your doctor.
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
