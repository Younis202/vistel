import { useState } from "react";
import { useParams, Link } from "wouter";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  ArrowLeft, ShieldAlert, CheckCircle2, AlertTriangle,
  Activity, FileText, Stethoscope, Brain, Eye, Bot,
  SendHorizonal, Download, ChevronRight, Loader2, Info,
  TrendingUp, CircleDot, Microscope
} from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

const DR_GRADE_COLORS = [
  "bg-green-100 text-green-800 border-green-200",
  "bg-yellow-100 text-yellow-800 border-yellow-200",
  "bg-orange-100 text-orange-800 border-orange-200",
  "bg-red-100 text-red-800 border-red-200",
  "bg-red-200 text-red-900 border-red-300",
];

const LESION_LABELS: Record<string, string> = {
  microaneurysm: "Microaneurysms",
  hemorrhage: "Dot/Blot Hemorrhages",
  hard_exudate: "Hard Exudates",
  soft_exudate: "Soft Exudates (CWS)",
  neovascularization: "Neovascularization",
  drusen: "Drusen Deposits",
  cotton_wool_spot: "Cotton-Wool Spots",
  venous_beading: "Venous Beading",
};

async function fetchCase(caseId: string) {
  const res = await fetch(`/api/retina/cases/${caseId}`);
  if (!res.ok) throw new Error("Case not found");
  return res.json();
}

async function downloadPdf(caseId: string) {
  const c = await fetch(`/api/retina/cases/${caseId}`);
  const data = await c.json();
  const fr = data.full_result || {};
  const dr = fr.dr_grading || {};

  const form = new FormData();
  const emptyPng = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
  const blob = await (await fetch(`data:image/png;base64,${emptyPng}`)).blob();
  form.append("file", blob, "placeholder.png");
  form.append("patient_id", data.patient_id || "UNKNOWN");

  const res = await fetch("/api/retina/report/pdf", { method: "POST", body: form });
  if (!res.ok) throw new Error("PDF generation failed");
  const pdfBlob = await res.blob();
  const url = URL.createObjectURL(pdfBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `retina_report_${caseId}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function RetinaAnalysisDetail() {
  const { id } = useParams<{ id: string }>();
  const caseId = id || "";

  const { data: caseData, isLoading, error } = useQuery({
    queryKey: ["retina-case", caseId],
    queryFn: () => fetchCase(caseId),
    enabled: !!caseId,
  });

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<{ role: "user" | "ai"; text: string; suggestion?: string }[]>([]);
  const [copilotError, setCopilotError] = useState<string | null>(null);

  const copilotMutation = useMutation({
    mutationFn: async (q: string) => {
      const res = await fetch("/api/retina/copilot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, case_id: caseId }),
      });
      if (!res.ok) throw new Error("Copilot request failed");
      return res.json();
    },
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: data.answer, suggestion: data.suggestion },
      ]);
    },
    onError: (err: any) => {
      setCopilotError(err.message);
    },
  });

  const handleAsk = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    const q = question.trim();
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setQuestion("");
    setCopilotError(null);
    copilotMutation.mutate(q);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-32">
        <div className="text-center space-y-3">
          <Loader2 className="h-10 w-10 text-primary animate-spin mx-auto" />
          <p className="text-slate-500">Loading diagnostic report...</p>
        </div>
      </div>
    );
  }

  if (error || !caseData) {
    return (
      <div className="max-w-2xl mx-auto p-12 text-center bg-red-50 rounded-3xl border border-red-100 mt-12">
        <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-red-900 mb-2">Report Not Found</h2>
        <p className="text-red-700">Case {caseId} could not be loaded.</p>
        <Link href="/patients">
          <button className="mt-6 px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700">
            Go Back
          </button>
        </Link>
      </div>
    );
  }

  const result = caseData.full_result || {};
  const dr = result.dr_grading || {};
  const amd = result.amd || {};
  const glaucoma = result.glaucoma || {};
  const quality = result.quality || {};
  const lesions = result.lesions || {};
  const report = result.report || {};
  const explain = result.explainability || {};

  const presentLesions = Object.entries(lesions).filter(([, v]: any) => v.present);
  const absentLesions = Object.entries(lesions).filter(([, v]: any) => !v.present);
  const riskLevel = caseData.risk_level || (dr.refer ? "high" : "low");

  const riskColors: Record<string, string> = {
    urgent: "bg-red-100 text-red-800 border-red-300",
    high: "bg-red-100 text-red-800 border-red-300",
    moderate: "bg-orange-100 text-orange-800 border-orange-200",
    low: "bg-green-100 text-green-800 border-green-200",
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-12">
      {/* Breadcrumb */}
      <div className="flex items-center justify-between">
        <Link href="/patients">
          <button className="flex items-center text-sm font-medium text-slate-500 hover:text-slate-900 transition-colors">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to Patients
          </button>
        </Link>
        <button
          onClick={() => downloadPdf(caseId)}
          className="px-4 py-2 bg-white border border-slate-200 shadow-sm rounded-lg text-sm font-semibold hover:bg-slate-50 flex items-center gap-2"
        >
          <Download className="h-4 w-4" />
          Export PDF
        </button>
      </div>

      {/* Header */}
      <div className="bg-white rounded-3xl p-6 md:p-8 shadow-sm border border-slate-200">
        <div className="flex flex-col md:flex-row justify-between gap-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-2xl font-display font-bold text-slate-900">Diagnostic Report</h1>
              <span className="px-3 py-1 rounded-full text-xs font-bold bg-primary/10 text-primary border border-primary/20">
                Retina-GPT v2
              </span>
              {result.model_version?.includes("demo") && (
                <span className="px-3 py-1 rounded-full text-xs font-bold bg-amber-100 text-amber-700 border border-amber-200">
                  Demo Mode
                </span>
              )}
            </div>
            <p className="text-slate-500 flex items-center gap-2 text-sm">
              <span className="font-medium text-slate-700">Patient ID: {caseData.patient_id}</span>
              <span>•</span>
              <span>Case: {caseData.id}</span>
              {caseData.created_at && (
                <>
                  <span>•</span>
                  <span>{format(new Date(caseData.created_at), "MMM dd, yyyy HH:mm")}</span>
                </>
              )}
            </p>
          </div>

          <div className="flex items-center gap-4">
            {dr.refer && (
              <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 px-4 py-2 rounded-xl text-sm font-bold">
                <ShieldAlert className="h-4 w-4" />
                Referral Recommended
              </div>
            )}
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Risk Level</p>
              <span className={cn("px-4 py-1.5 rounded-full text-sm font-bold border inline-flex items-center gap-2 uppercase tracking-wide", riskColors[riskLevel] || riskColors.low)}>
                {riskLevel === "urgent" || riskLevel === "high" ? <ShieldAlert className="h-4 w-4" /> : <CheckCircle2 className="h-4 w-4" />}
                {riskLevel}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Grad-CAM Image */}
          <div className="bg-slate-900 rounded-3xl overflow-hidden shadow-lg border border-slate-800">
            <div className="p-4 border-b border-slate-700 flex items-center gap-2">
              <Eye className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold text-white">Grad-CAM Explainability</span>
            </div>
            {explain.gradcam_image ? (
              <img
                src={`data:image/png;base64,${explain.gradcam_image}`}
                alt="Grad-CAM"
                className="w-full aspect-square object-cover"
              />
            ) : (
              <div className="w-full aspect-square flex flex-col items-center justify-center text-slate-500 p-8">
                <Eye className="h-12 w-12 mb-2 opacity-30" />
                <p className="text-sm">No explainability image available</p>
              </div>
            )}
            {explain.gradcam_image && (
              <div className="p-3 text-xs text-slate-400 text-center">
                Hot regions indicate areas driving the AI diagnosis
              </div>
            )}
          </div>

          {/* Quality Score */}
          <div className="bg-white rounded-3xl p-5 shadow-sm border border-slate-200">
            <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4 text-primary" />
              Image Quality
            </h3>
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-500 text-sm">Quality Score</span>
              <span className="font-bold text-slate-900">{((quality.score || 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className={cn("h-full rounded-full", quality.adequate ? "bg-green-500" : "bg-amber-500")}
                style={{ width: `${(quality.score || 0) * 100}%` }}
              />
            </div>
            <p className={cn("text-xs mt-2 font-medium", quality.adequate ? "text-green-600" : "text-amber-600")}>
              {quality.adequate ? "✓ Adequate for clinical analysis" : "⚠ Suboptimal — consider retaking"}
            </p>
          </div>

          {/* Processing Details */}
          <div className="bg-white rounded-3xl p-5 shadow-sm border border-slate-200">
            <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4 text-primary" />
              Processing Details
            </h3>
            <div className="space-y-2 text-sm">
              {[
                ["Inference Time", `${result.inference_time_ms?.toFixed(0) || "N/A"} ms`],
                ["Model", result.model_version || "Retina-GPT"],
                ["Pathologies Screened", "13"],
                ["Image File", caseData.image_name || "N/A"],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between py-1.5 border-b border-slate-100 last:border-0">
                  <span className="text-slate-500">{k}</span>
                  <span className="font-medium text-slate-900 text-right max-w-[60%] truncate">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="lg:col-span-2 space-y-6">
          {/* Primary Diagnoses */}
          <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-200">
            <h3 className="text-xl font-display font-bold text-slate-900 mb-5 flex items-center gap-2">
              <Stethoscope className="h-5 w-5 text-primary" />
              Primary Diagnoses
            </h3>

            <div className="space-y-4">
              {/* DR Grading */}
              <div className={cn("p-4 rounded-2xl border", DR_GRADE_COLORS[dr.grade] || "bg-slate-50 border-slate-200")}>
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <CircleDot className="h-4 w-4" />
                      <span className="text-xs font-bold uppercase tracking-wider opacity-70">Diabetic Retinopathy</span>
                    </div>
                    <p className="text-lg font-bold">{dr.label || "Unknown"}</p>
                    <p className="text-sm opacity-70 mt-0.5">Grade {dr.grade ?? "N/A"} of 4</p>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-2xl font-bold">{((dr.confidence || 0) * 100).toFixed(0)}%</div>
                    <div className="text-xs opacity-70">confidence</div>
                  </div>
                </div>
                {dr.probabilities && dr.probabilities.length > 0 && (
                  <div className="mt-3 flex gap-1">
                    {dr.probabilities.map((p: number, i: number) => (
                      <div key={i} className="flex-1 text-center">
                        <div className="h-1 rounded-full bg-current opacity-30 overflow-hidden">
                          <div className="h-full bg-current" style={{ width: `${p * 100}%` }} />
                        </div>
                        <div className="text-xs mt-1 opacity-60">G{i}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* AMD */}
              <div className={cn("p-4 rounded-2xl border", amd.stage > 0 ? "bg-orange-50 border-orange-200" : "bg-slate-50 border-slate-200")}>
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingUp className="h-4 w-4 text-orange-600" />
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Age-Related Macular Degeneration</span>
                    </div>
                    <p className="text-lg font-bold text-slate-900">{amd.label || "No AMD"}</p>
                    <p className="text-sm text-slate-500">Stage {amd.stage ?? 0} of 3</p>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-2xl font-bold text-slate-900">{((amd.confidence || 0) * 100).toFixed(0)}%</div>
                    <div className="text-xs text-slate-400">confidence</div>
                  </div>
                </div>
              </div>

              {/* Glaucoma */}
              <div className={cn("p-4 rounded-2xl border", glaucoma.suspect ? "bg-purple-50 border-purple-200" : "bg-slate-50 border-slate-200")}>
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Eye className="h-4 w-4 text-purple-600" />
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Glaucoma Screening</span>
                    </div>
                    <p className="text-lg font-bold text-slate-900">
                      {glaucoma.suspect ? "Glaucoma Suspected" : "No Glaucoma Suspicion"}
                    </p>
                    <p className="text-sm text-slate-500">
                      Cup-to-disc ratio: <span className="font-semibold">{glaucoma.cup_disc_ratio?.toFixed(2) || "N/A"}</span>
                      {" "}(normal &lt; 0.6)
                    </p>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-2xl font-bold text-slate-900">{((glaucoma.confidence || 0) * 100).toFixed(0)}%</div>
                    <div className="text-xs text-slate-400">confidence</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Lesion Detection */}
          <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-200">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-xl font-display font-bold text-slate-900 flex items-center gap-2">
                <Microscope className="h-5 w-5 text-primary" />
                Lesion Detection
              </h3>
              <span className="text-sm font-medium text-slate-500">
                {presentLesions.length} detected
              </span>
            </div>

            <div className="space-y-3">
              {[...presentLesions, ...absentLesions].map(([key, val]: any) => (
                <div
                  key={key}
                  className={cn(
                    "flex items-center justify-between p-3 rounded-xl border",
                    val.present ? "bg-white border-slate-200 shadow-sm" : "bg-slate-50 border-transparent opacity-60"
                  )}
                >
                  <div className="flex items-center gap-3">
                    {val.present ? (
                      <div className="h-7 w-7 rounded-full bg-red-100 text-red-600 flex items-center justify-center shrink-0">
                        <ShieldAlert className="h-3.5 w-3.5" />
                      </div>
                    ) : (
                      <div className="h-7 w-7 rounded-full bg-slate-200 text-slate-400 flex items-center justify-center shrink-0">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                      </div>
                    )}
                    <span className={cn("font-medium text-sm", val.present ? "text-slate-900" : "text-slate-500")}>
                      {LESION_LABELS[key] || key}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    {val.present && (
                      <span className="text-xs font-bold text-red-700 bg-red-50 border border-red-200 px-2 py-0.5 rounded">
                        DETECTED
                      </span>
                    )}
                    <div className="flex items-center gap-1.5 bg-slate-100 px-2.5 py-1 rounded-lg">
                      <span className="text-xs font-semibold text-slate-500">CONF.</span>
                      <span className="text-sm font-bold text-slate-900">{((val.probability || 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Clinical Report */}
          <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-200">
            <h3 className="text-xl font-display font-bold text-slate-900 mb-4 flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              Clinical Report
            </h3>

            {report.structured_findings && (
              <div className="bg-slate-50 border border-slate-100 rounded-xl p-4 mb-4">
                <pre className="whitespace-pre-wrap text-sm text-slate-700 font-sans leading-relaxed">
                  {report.structured_findings}
                </pre>
              </div>
            )}

            {report.recommendation && (
              <div className="bg-blue-50 border border-blue-100 p-4 rounded-xl text-blue-900 flex items-start gap-3">
                <ChevronRight className="h-5 w-5 text-blue-600 shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-bold text-blue-600 uppercase tracking-wider mb-1">Recommendation</p>
                  <p className="leading-relaxed text-sm">{report.recommendation}</p>
                </div>
              </div>
            )}
          </div>

          {/* AI Copilot */}
          <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-200">
            <h3 className="text-xl font-display font-bold text-slate-900 mb-1 flex items-center gap-2">
              <Bot className="h-5 w-5 text-primary" />
              AI Copilot
            </h3>
            <p className="text-slate-500 text-sm mb-5">
              Ask the AI clinical questions about this scan — referral urgency, lesion explanations, confidence levels.
            </p>

            <div className="bg-slate-50 rounded-2xl p-4 min-h-[160px] max-h-[320px] overflow-y-auto mb-4 space-y-3">
              {messages.length === 0 && (
                <div className="text-center text-slate-400 text-sm py-6">
                  <Brain className="h-8 w-8 mx-auto mb-2 opacity-30" />
                  Ask a clinical question to get started
                </div>
              )}
              {messages.map((m, i) => (
                <div key={i} className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}>
                  <div className={cn(
                    "max-w-[85%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed",
                    m.role === "user"
                      ? "bg-primary text-white"
                      : "bg-white border border-slate-200 text-slate-800"
                  )}>
                    {m.text}
                    {m.suggestion && m.role === "ai" && (
                      <p className="text-xs mt-2 opacity-60 italic">{m.suggestion}</p>
                    )}
                  </div>
                </div>
              ))}
              {copilotMutation.isPending && (
                <div className="flex justify-start">
                  <div className="bg-white border border-slate-200 px-4 py-2.5 rounded-2xl">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  </div>
                </div>
              )}
            </div>

            {copilotError && (
              <p className="text-red-600 text-xs mb-3 flex items-center gap-1">
                <Info className="h-3 w-3" />
                {copilotError}
              </p>
            )}

            <form onSubmit={handleAsk} className="flex gap-2">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder='e.g. "Should I refer this patient?" or "Explain the DR grade"'
                className="flex-1 border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
              />
              <button
                type="submit"
                disabled={!question.trim() || copilotMutation.isPending}
                className="px-4 py-2.5 bg-primary text-white rounded-xl hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
              >
                <SendHorizonal className="h-4 w-4" />
              </button>
            </form>

            <div className="flex flex-wrap gap-2 mt-3">
              {[
                "Should I refer this patient?",
                "What lesions were detected?",
                "Explain the DR grade",
                "How confident is the AI?",
              ].map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setQuestion(q)}
                  className="text-xs px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
