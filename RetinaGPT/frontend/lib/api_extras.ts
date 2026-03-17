// Append to existing lib/api.ts

export const askCopilot = async (
  case_id: string,
  question: string
): Promise<{
  question: string; answer: string; confidence: number;
  intents: string[]; sources: string[]; suggestion: string;
}> => (await http.post('/copilot', { case_id, question })).data

export const createReferral = async (body: {
  case_id: string; patient_id: string; referring_dr?: string;
  specialist?: string; clinic?: string; reason?: string;
  urgency?: string; notes?: string;
}) => (await http.post('/referrals', body)).data

export const getReferrals = async (params?: {
  patient_id?: string; case_id?: string; status?: string;
}) => (await http.get('/referrals', { params })).data

export const getReferralStats = async () => (await http.get('/referrals/stats')).data

export const updateReferral = async (
  id: string, body: { status: string; notes?: string; outcome?: string }
) => (await http.patch(`/referrals/${id}`, body)).data

export const createPassport = async (body: {
  case_id: string; patient_id: string; expires_days?: number;
}): Promise<{ token: string; share_url: string; views: number }> =>
  (await http.post('/passport', body)).data

export const getPassport = async (token: string) =>
  (await http.get(`/passport/${token}`)).data

export const revokePassport = async (token: string) =>
  http.delete(`/passport/${token}`)
