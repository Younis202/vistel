'use client'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import axios from 'axios'

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const STATUSES = [
  { key: 'pending',      label: 'Pending',      color: '#B8860B', bg: '#FBF5E6' },
  { key: 'sent',         label: 'Sent',          color: '#185FA5', bg: '#E6F1FB' },
  { key: 'acknowledged', label: 'Acknowledged',  color: '#534AB7', bg: '#EEEDFE' },
  { key: 'seen',         label: 'Seen',          color: '#0F6E56', bg: '#E1F5EE' },
  { key: 'completed',    label: 'Completed',     color: '#1A3A2A', bg: '#EBF3EE' },
  { key: 'cancelled',    label: 'Cancelled',     color: '#7A7A72', bg: '#F1EFE8' },
]

const URGENCY = [
  { key: 'urgent',   label: 'Urgent',   color: '#8B2020', bg: '#FBF0EE' },
  { key: 'priority', label: 'Priority', color: '#B8860B', bg: '#FBF5E6' },
  { key: 'routine',  label: 'Routine',  color: '#2C5C42', bg: '#EBF3EE' },
]

function StatusBadge({ status }: { status: string }) {
  const s = STATUSES.find(x => x.key === status) || STATUSES[0]
  return (
    <span style={{
      fontSize: 10, padding: '3px 8px', borderRadius: 4, fontFamily: 'var(--mono)',
      fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase',
      background: s.bg, color: s.color, border: `1px solid ${s.color}22`,
    }}>{s.label}</span>
  )
}

function UrgencyBadge({ urgency }: { urgency: string }) {
  const u = URGENCY.find(x => x.key === urgency) || URGENCY[2]
  return (
    <span style={{
      fontSize: 10, padding: '2px 7px', borderRadius: 4, fontFamily: 'var(--mono)',
      background: u.bg, color: u.color,
    }}>{u.label}</span>
  )
}

export default function ReferralPanel({
  caseId, patientId, drGrade, drLabel, onClose
}: {
  caseId: string; patientId: string; drGrade?: number; drLabel?: string; onClose: () => void;
}) {
  const [mode, setMode] = useState<'create' | 'list'>('create')
  const [referrals, setReferrals] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [form, setForm] = useState({
    referring_dr: '', specialist: '', clinic: '',
    reason: '', urgency: 'routine', notes: ''
  })

  const loadReferrals = async () => {
    const { data } = await axios.get(`${BASE}/referrals`, { params: { case_id: caseId } })
    setReferrals(data.referrals || [])
    setMode('list')
  }

  const submit = async () => {
    if (!form.specialist) return toast.error('Enter specialist name')
    setLoading(true)
    try {
      await axios.post(`${BASE}/referrals`, { case_id: caseId, patient_id: patientId, ...form })
      toast.success('Referral created')
      loadReferrals()
    } catch { toast.error('Failed to create referral') }
    finally { setLoading(false) }
  }

  const advance = async (id: string, current: string) => {
    const order = STATUSES.map(s => s.key)
    const next  = order[order.indexOf(current) + 1]
    if (!next || next === 'cancelled') return
    try {
      await axios.patch(`${BASE}/referrals/${id}`, { status: next })
      toast.success(`Marked as ${next}`)
      loadReferrals()
    } catch { toast.error('Update failed') }
  }

  const cancel = async (id: string) => {
    try {
      await axios.patch(`${BASE}/referrals/${id}`, { status: 'cancelled' })
      toast.success('Referral cancelled')
      loadReferrals()
    } catch { toast.error('Could not cancel') }
  }

  return (
    <div style={{
      background: 'var(--paper)', border: '1px solid var(--paper3)',
      borderRadius: 12, overflow: 'hidden',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '14px 16px', borderBottom: '1px solid var(--paper3)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink)' }}>Referral</div>
          {drGrade !== undefined && drGrade >= 2 && (
            <span style={{ fontSize: 10, padding: '2px 7px', borderRadius: 4, background: 'var(--red-pale)', color: 'var(--red)', fontFamily: 'var(--mono)' }}>
              Grade {drGrade} — Refer
            </span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={loadReferrals} style={{ fontSize: 11, background: 'none', border: '1px solid var(--paper3)', borderRadius: 5, padding: '4px 10px', cursor: 'pointer', color: 'var(--ink3)', fontFamily: 'var(--sans)' }}>
            History
          </button>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--ink3)', fontSize: 18, lineHeight: 1, padding: 2 }}>×</button>
        </div>
      </div>

      <div style={{ padding: 16 }}>
        <AnimatePresence mode="wait">
          {mode === 'create' ? (
            <motion.div key="create" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                {[
                  { id: 'referring_dr', label: 'Referring Doctor', placeholder: 'Dr. Ahmed' },
                  { id: 'specialist',   label: 'Specialist *',     placeholder: 'Retina specialist' },
                  { id: 'clinic',       label: 'Clinic',           placeholder: 'Ophthalmology Clinic' },
                ].map(f => (
                  <div key={f.id} style={{ gridColumn: f.id === 'clinic' ? '1/-1' : undefined }}>
                    <div style={{ fontSize: 10, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ink3)', marginBottom: 5 }}>
                      {f.label}
                    </div>
                    <input className="field" placeholder={f.placeholder}
                      value={(form as any)[f.id]} onChange={e => setForm(x => ({ ...x, [f.id]: e.target.value }))} />
                  </div>
                ))}
              </div>

              <div>
                <div style={{ fontSize: 10, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ink3)', marginBottom: 6 }}>Urgency</div>
                <div style={{ display: 'flex', gap: 7 }}>
                  {URGENCY.map(u => (
                    <button key={u.key} onClick={() => setForm(x => ({ ...x, urgency: u.key }))} style={{
                      flex: 1, padding: '7px 0', border: `1px solid ${form.urgency === u.key ? u.color : 'var(--paper3)'}`,
                      borderRadius: 6, background: form.urgency === u.key ? u.bg : 'transparent',
                      color: form.urgency === u.key ? u.color : 'var(--ink3)',
                      fontSize: 12, cursor: 'pointer', fontFamily: 'var(--sans)', fontWeight: form.urgency === u.key ? 500 : 400,
                      transition: 'all 0.12s',
                    }}>{u.label}</button>
                  ))}
                </div>
              </div>

              <div>
                <div style={{ fontSize: 10, fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ink3)', marginBottom: 5 }}>
                  Reason / Notes
                </div>
                <textarea className="field" rows={2} placeholder="Moderate DR with microaneurysms detected…"
                  value={form.reason} onChange={e => setForm(x => ({ ...x, reason: e.target.value }))}
                  style={{ resize: 'none', fontFamily: 'var(--sans)', lineHeight: 1.5 }} />
              </div>

              <button className="btn btn-forest btn-lg" onClick={submit} disabled={loading}>
                {loading ? 'Creating…' : 'Create Referral'}
              </button>
            </motion.div>
          ) : (
            <motion.div key="list" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              {referrals.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--ink3)', fontSize: 13 }}>
                  No referrals yet.
                  <button onClick={() => setMode('create')} style={{ display: 'block', margin: '8px auto 0', background: 'none', border: 'none', color: 'var(--forest3)', fontSize: 12, cursor: 'pointer', fontFamily: 'var(--sans)' }}>
                    Create first referral →
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {referrals.map(r => (
                    <div key={r.id} style={{ border: '1px solid var(--paper3)', borderRadius: 8, padding: 14 }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <StatusBadge status={r.status} />
                          <UrgencyBadge urgency={r.urgency} />
                        </div>
                        <span style={{ fontSize: 10, color: 'var(--ink3)', fontFamily: 'var(--mono)' }}>
                          {r.created_at?.slice(0, 10)}
                        </span>
                      </div>
                      <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink)' }}>{r.specialist}</div>
                      {r.clinic && <div style={{ fontSize: 11, color: 'var(--ink3)', marginTop: 1 }}>{r.clinic}</div>}
                      {r.reason && <div style={{ fontSize: 12, color: 'var(--ink2)', marginTop: 6, lineHeight: 1.4 }}>{r.reason}</div>}

                      {!['completed','cancelled'].includes(r.status) && (
                        <div style={{ display: 'flex', gap: 7, marginTop: 10 }}>
                          <button onClick={() => advance(r.id, r.status)} style={{
                            flex: 1, padding: '6px 0', border: '1px solid var(--forest)', borderRadius: 5,
                            background: 'var(--forest-pale)', color: 'var(--forest2)', fontSize: 11,
                            cursor: 'pointer', fontFamily: 'var(--sans)',
                          }}>
                            Mark next →
                          </button>
                          <button onClick={() => cancel(r.id)} style={{
                            padding: '6px 12px', border: '1px solid var(--paper3)', borderRadius: 5,
                            background: 'transparent', color: 'var(--ink3)', fontSize: 11,
                            cursor: 'pointer', fontFamily: 'var(--sans)',
                          }}>
                            Cancel
                          </button>
                        </div>
                      )}
                    </div>
                  ))}
                  <button onClick={() => setMode('create')} style={{
                    width: '100%', padding: '9px', border: '1.5px dashed var(--paper3)',
                    borderRadius: 7, background: 'none', color: 'var(--ink3)',
                    fontSize: 12, cursor: 'pointer', fontFamily: 'var(--sans)',
                  }}>+ New referral</button>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
