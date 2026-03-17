'use client'
import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

interface Message {
  role: 'user' | 'assistant'
  text: string
  confidence?: number
  suggestion?: string
}

const QUICK = [
  "Should I refer this patient?",
  "What lesions are present?",
  "Explain why you graded this level.",
  "How confident are you?",
  "Is image quality adequate?",
]

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function Copilot({ caseId, onClose }: { caseId: string; onClose: () => void }) {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', text: 'I have analysed this scan. Ask me anything about the findings, grading, or clinical decisions.' }
  ])
  const [input,   setInput]   = useState('')
  const [loading, setLoading] = useState(false)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const ask = async (q: string) => {
    if (!q.trim() || loading) return
    setMessages(m => [...m, { role: 'user', text: q }])
    setInput('')
    setLoading(true)
    try {
      const { data } = await axios.post(`${BASE}/copilot`, { case_id: caseId, question: q })
      setMessages(m => [...m, {
        role: 'assistant',
        text: data.answer,
        confidence: data.confidence,
        suggestion: data.suggestion,
      }])
    } catch {
      setMessages(m => [...m, { role: 'assistant', text: 'Could not reach the AI. Is the API running?' }])
    } finally { setLoading(false) }
  }

  return (
    <div style={{
      background: 'var(--paper)', border: '1px solid var(--paper3)', borderRadius: 12,
      display: 'flex', flexDirection: 'column', height: 480, overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '14px 16px', borderBottom: '1px solid var(--paper3)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 28, height: 28, borderRadius: '50%',
            background: 'var(--forest)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#FAFAF7" strokeWidth="1.5">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--ink)' }}>AI Copilot</div>
            <div style={{ fontSize: 10, color: 'var(--ink3)', fontFamily: 'var(--mono)' }}>
              Case {caseId.slice(0, 8)}
            </div>
          </div>
        </div>
        <button onClick={onClose} style={{
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--ink3)', fontSize: 18, lineHeight: 1, padding: 4,
        }}>×</button>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.map((m, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}
          >
            <div style={{
              maxWidth: '82%',
              background: m.role === 'user' ? 'var(--forest)' : 'var(--paper2)',
              color: m.role === 'user' ? '#FAFAF7' : 'var(--ink2)',
              borderRadius: m.role === 'user' ? '10px 10px 2px 10px' : '10px 10px 10px 2px',
              padding: '10px 13px',
              fontSize: 13,
              lineHeight: 1.55,
              border: m.role === 'assistant' ? '1px solid var(--paper3)' : 'none',
            }}>
              {m.text}
              {m.confidence !== undefined && (
                <div style={{
                  marginTop: 7, paddingTop: 7, borderTop: '1px solid rgba(0,0,0,0.08)',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}>
                  <div style={{
                    flex: 1, height: 2, background: 'var(--paper3)', borderRadius: 1, overflow: 'hidden'
                  }}>
                    <div style={{ height: '100%', width: `${m.confidence * 100}%`, background: 'var(--forest3)' }} />
                  </div>
                  <span style={{ fontSize: 10, color: 'var(--ink3)', fontFamily: 'var(--mono)', whiteSpace: 'nowrap' }}>
                    {Math.round(m.confidence * 100)}% confidence
                  </span>
                </div>
              )}
              {m.suggestion && (
                <button
                  onClick={() => ask(m.suggestion!)}
                  style={{
                    marginTop: 8, display: 'block', width: '100%', textAlign: 'left',
                    background: 'rgba(44,92,66,0.08)', border: '1px solid rgba(44,92,66,0.15)',
                    borderRadius: 6, padding: '6px 9px', fontSize: 11, color: 'var(--forest2)',
                    cursor: 'pointer', fontFamily: 'var(--sans)',
                  }}
                >
                  → {m.suggestion}
                </button>
              )}
            </div>
          </motion.div>
        ))}

        {loading && (
          <div style={{ display: 'flex', gap: 4, padding: '4px 2px' }}>
            {[0,1,2].map(i => (
              <motion.div key={i} style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--ink3)' }}
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }} />
            ))}
          </div>
        )}
        <div ref={bottom} />
      </div>

      {/* Quick prompts */}
      {messages.length <= 1 && (
        <div style={{ padding: '0 16px 10px', display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {QUICK.map(q => (
            <button key={q} onClick={() => ask(q)} style={{
              padding: '5px 10px', border: '1px solid var(--paper3)', borderRadius: 4,
              background: 'var(--paper2)', color: 'var(--ink2)', fontSize: 11,
              cursor: 'pointer', fontFamily: 'var(--sans)', transition: 'border-color 0.12s',
            }}
              onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--forest3)')}
              onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--paper3)')}
            >{q}</button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{ padding: '0 12px 12px', borderTop: '1px solid var(--paper3)', paddingTop: 12 }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && ask(input)}
            placeholder="Ask about this scan…"
            disabled={loading}
            style={{
              flex: 1, border: '1px solid var(--paper3)', borderRadius: 7,
              padding: '9px 12px', fontSize: 13, background: 'var(--paper)',
              color: 'var(--ink)', fontFamily: 'var(--sans)', outline: 'none',
            }}
          />
          <button onClick={() => ask(input)} disabled={!input.trim() || loading}
            style={{
              padding: '9px 14px', background: 'var(--forest)', border: 'none',
              borderRadius: 7, color: '#FAFAF7', fontSize: 13, cursor: 'pointer',
              fontFamily: 'var(--sans)', fontWeight: 500, opacity: (!input.trim() || loading) ? 0.4 : 1,
            }}>
            Ask
          </button>
        </div>
      </div>
    </div>
  )
}
