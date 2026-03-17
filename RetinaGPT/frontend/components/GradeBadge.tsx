const CLS = ['g0','g1','g2','g3','g4'] as const
const LABELS = ['No DR','Mild NPDR','Moderate NPDR','Severe NPDR','Proliferative DR']

export default function GradeBadge({
  grade, showRefer = false, size = 'sm'
}: { grade: number; showRefer?: boolean; size?: 'sm' | 'md' }) {
  const g = Math.min(Math.max(grade, 0), 4)
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span className={`grade-pill ${CLS[g]}`} style={size === 'md' ? { fontSize: 13, padding: '5px 11px' } : {}}>
        {size === 'md' ? `Grade ${g} — ${LABELS[g]}` : LABELS[g]}
      </span>
      {showRefer && g >= 2 && <span className="refer-tag">Refer</span>}
    </span>
  )
}
