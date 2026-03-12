import { useParams, Link } from "wouter";
import { useGetAnalysis, useGetPatient } from "@workspace/api-client-react";
import { 
  ArrowLeft, ShieldAlert, CheckCircle2, AlertTriangle, 
  Activity, Image as ImageIcon, FileText, Stethoscope, ChevronRight 
} from "lucide-react";
import { format } from "date-fns";
import { formatRiskColor, formatSeverityColor, cn } from "@/lib/utils";

export default function AnalysisDetail() {
  const { id } = useParams<{ id: string }>();
  const analysisId = parseInt(id || "0");

  const { data: analysis, isLoading: aLoading, error } = useGetAnalysis(analysisId);
  const { data: patient } = useGetPatient(analysis?.patientId || 0, {
    query: { enabled: !!analysis?.patientId }
  });

  if (aLoading) {
    return <div className="p-8 text-center text-slate-500 animate-pulse">Loading diagnostic report...</div>;
  }

  if (error || !analysis) {
    return (
      <div className="max-w-2xl mx-auto p-12 text-center bg-red-50 rounded-3xl border border-red-100 mt-12">
        <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-red-900 mb-2">Report Not Found</h2>
        <p className="text-red-700">The requested analysis report could not be loaded.</p>
        <Link href="/patients">
          <button className="mt-6 px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700">Go Back</button>
        </Link>
      </div>
    );
  }

  // Sort diseases: detected first, then by severity, then confidence
  const severityScore: Record<string, number> = { severe: 4, moderate: 3, mild: 2, none: 1 };
  const sortedDiseases = [...analysis.diseases].sort((a, b) => {
    if (a.detected !== b.detected) return a.detected ? -1 : 1;
    if (severityScore[a.severity] !== severityScore[b.severity]) return severityScore[b.severity] - severityScore[a.severity];
    return b.confidence - a.confidence;
  });

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-12">
      <div className="flex items-center justify-between">
        <Link href={`/patients/${analysis.patientId}`}>
          <button className="flex items-center text-sm font-medium text-slate-500 hover:text-slate-900 transition-colors">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to Patient
          </button>
        </Link>
        <button className="px-4 py-2 bg-white border border-slate-200 shadow-sm rounded-lg text-sm font-semibold hover:bg-slate-50 flex items-center gap-2">
          <FileText className="h-4 w-4" />
          Export PDF
        </button>
      </div>

      {/* Header Card */}
      <div className="bg-white rounded-3xl p-6 md:p-8 shadow-sm border border-slate-200">
        <div className="flex flex-col md:flex-row justify-between gap-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-2xl font-display font-bold text-slate-900">Diagnostic Report</h1>
              <span className="px-3 py-1 rounded-full text-xs font-bold bg-primary/10 text-primary border border-primary/20">
                AI Verified
              </span>
            </div>
            <p className="text-slate-500 flex items-center gap-2">
              <span className="font-medium text-slate-700">{patient?.name || `Patient #${analysis.patientId}`}</span>
              <span>•</span>
              {format(new Date(analysis.createdAt), 'MMMM dd, yyyy - HH:mm')}
            </p>
          </div>

          <div className="flex items-center gap-4 text-right">
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Overall Risk Level</p>
              <div className={cn("px-4 py-1.5 rounded-full text-sm font-bold border inline-flex items-center gap-2 uppercase tracking-wide", formatRiskColor(analysis.overallRisk))}>
                {analysis.overallRisk === 'critical' || analysis.overallRisk === 'high' ? <ShieldAlert className="h-4 w-4" /> : <CheckCircle2 className="h-4 w-4" />}
                {analysis.overallRisk}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Image & Info */}
        <div className="space-y-6">
          <div className="bg-slate-900 rounded-3xl overflow-hidden shadow-lg border border-slate-800 relative group">
            <div className="absolute top-4 left-4 z-10 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 text-white text-xs font-semibold tracking-wide uppercase">
              {analysis.eyeSide} Eye
            </div>
            {analysis.imageUrl ? (
              <img src={analysis.imageUrl} alt="Fundus" className="w-full aspect-square object-cover" />
            ) : (
              <div className="w-full aspect-square flex flex-col items-center justify-center text-slate-500">
                <ImageIcon className="h-12 w-12 mb-2 opacity-50" />
                <p>Image not stored</p>
              </div>
            )}
            
            {/* Image Quality Overlay */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-6 pt-12">
              <div className="flex justify-between items-end">
                <div>
                  <p className="text-slate-300 text-xs font-medium mb-1">Image Quality Score</p>
                  <p className="text-white font-display font-bold text-2xl">
                    {(analysis.imageQualityScore * 100).toFixed(0)}<span className="text-slate-400 text-lg">%</span>
                  </p>
                </div>
                {analysis.imageQualityOk ? (
                  <div className="bg-green-500/20 text-green-400 border border-green-500/30 px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 backdrop-blur-md">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Optimal
                  </div>
                ) : (
                  <div className="bg-amber-500/20 text-amber-400 border border-amber-500/30 px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 backdrop-blur-md">
                    <AlertTriangle className="h-3.5 w-3.5" /> Suboptimal
                  </div>
                )}
              </div>
              <div className="w-full h-1.5 bg-slate-700 rounded-full mt-3 overflow-hidden">
                <div 
                  className={cn("h-full rounded-full", analysis.imageQualityOk ? "bg-green-500" : "bg-amber-500")}
                  style={{ width: `${analysis.imageQualityScore * 100}%` }}
                />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-200">
            <h3 className="text-lg font-display font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Processing Details
            </h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">Analysis Time</span>
                <span className="font-medium text-slate-900">{analysis.analysisTime.toFixed(2)}s</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">Models Run</span>
                <span className="font-medium text-slate-900">3 (Q-Check, Lesion, Classify)</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-slate-500">Diseases Screened</span>
                <span className="font-medium text-slate-900">13 Pathologies</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Findings */}
        <div className="lg:col-span-2 space-y-6">
          
          <div className="bg-white rounded-3xl p-6 md:p-8 shadow-sm border border-slate-200">
            <h3 className="text-xl font-display font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Stethoscope className="h-6 w-6 text-primary" />
              Clinical Summary
            </h3>
            <p className="text-slate-700 leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-100">
              {analysis.summary}
            </p>

            <h4 className="font-semibold text-slate-900 mt-6 mb-2">Recommendations:</h4>
            <div className="bg-blue-50 border border-blue-100 p-4 rounded-xl text-blue-900 flex items-start gap-3">
              <CheckCircle2 className="h-5 w-5 text-blue-600 shrink-0 mt-0.5" />
              <p className="leading-relaxed">{analysis.recommendations}</p>
            </div>
          </div>

          <div className="bg-white rounded-3xl p-6 md:p-8 shadow-sm border border-slate-200">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-display font-bold text-slate-900">Pathology Detection</h3>
              <span className="text-sm font-medium text-slate-500">
                {sortedDiseases.filter(d => d.detected).length} Detected
              </span>
            </div>

            <div className="space-y-4">
              {sortedDiseases.map((disease, idx) => (
                <div 
                  key={idx} 
                  className={cn(
                    "p-4 rounded-2xl border transition-all",
                    disease.detected ? "bg-white border-slate-200 shadow-sm" : "bg-slate-50 border-transparent opacity-60"
                  )}
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-3">
                    <div className="flex items-center gap-3">
                      {disease.detected ? (
                        <div className="h-8 w-8 rounded-full bg-red-100 text-red-600 flex items-center justify-center shrink-0">
                          <ShieldAlert className="h-4 w-4" />
                        </div>
                      ) : (
                        <div className="h-8 w-8 rounded-full bg-slate-200 text-slate-400 flex items-center justify-center shrink-0">
                          <CheckCircle2 className="h-4 w-4" />
                        </div>
                      )}
                      <h4 className={cn("font-bold text-lg", disease.detected ? "text-slate-900" : "text-slate-600")}>
                        {disease.name}
                      </h4>
                    </div>
                    <div className="flex items-center gap-3 self-start sm:self-auto ml-11 sm:ml-0">
                      {disease.detected && (
                        <span className={cn("px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wider", formatSeverityColor(disease.severity))}>
                          {disease.severity}
                        </span>
                      )}
                      <div className="flex items-center gap-2 bg-slate-100 px-3 py-1 rounded-lg">
                        <span className="text-xs font-semibold text-slate-500">CONF.</span>
                        <span className="text-sm font-bold text-slate-900">{(disease.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>

                  {disease.detected && disease.findings && (
                    <div className="ml-11 bg-slate-50 p-3 rounded-xl border border-slate-100 text-sm text-slate-700">
                      <span className="font-semibold text-slate-900 mr-2">Findings:</span>
                      {disease.findings}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
