import axios from 'axios'
import type { AnalysisResult, CaseStats, CaseEntry, SearchResult, ProgressionReport } from '@/types'

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const http = axios.create({ baseURL: BASE, timeout: 120000 })

export const checkHealth = async (): Promise<boolean> => {
  try { const r = await http.get('/health', { timeout: 3000 }); return !!r.data.status } catch { return false }
}

export const analyzeImage = async (
  file: File,
  opts: { explain?: boolean; segment?: boolean; patientId?: string } = {}
): Promise<AnalysisResult> => {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('explain',  String(opts.explain  ?? true))
  fd.append('segment', String(opts.segment ?? false))
  if (opts.patientId) fd.append('image_id', opts.patientId)
  return (await http.post('/analyze', fd)).data
}

export const getCaseStats = async (): Promise<CaseStats> => (await http.get('/cases/stats')).data

export const getCases = async (p?: {
  limit?: number; offset?: number; patient_id?: string
  dr_grade?: number; refer_only?: boolean
}): Promise<{ total: number; cases: CaseEntry[] }> => (await http.get('/cases', { params: p })).data

export const getCase = async (id: string) => (await http.get(`/cases/${id}`)).data

export const deleteCase = async (id: string) => http.delete(`/cases/${id}`)

export const searchSimilar = async (
  file: File, k = 10, dr_grade?: number
): Promise<{ results: SearchResult[]; index_size: number; search_time_ms: number }> => {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('k', String(k))
  if (dr_grade !== undefined) fd.append('dr_grade', String(dr_grade))
  return (await http.post('/search', fd)).data
}

export const downloadPDF = async (file: File, patientId: string): Promise<Blob> => {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('patient_id', patientId)
  return (await http.post('/report/pdf', fd, { responseType: 'blob' })).data
}

export const analyzeProgression = async (
  patientId: string,
  visits: Array<{ visit_date: string; image_b64: string }>
): Promise<ProgressionReport> => (await http.post('/progression', { patient_id: patientId, visits })).data

export const b64 = (s: string | null) => s ? `data:image/png;base64,${s}` : null
