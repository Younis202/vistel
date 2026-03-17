import { Link } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { useListPatients } from "@workspace/api-client-react";
import {
  Users,
  Activity,
  FileText,
  ArrowRight,
  TrendingUp,
  Clock,
  ShieldAlert,
  ScanEye,
  CheckCircle2,
} from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { format, subDays } from "date-fns";
import { cn } from "@/lib/utils";

const MOCK_CHART_DATA = Array.from({ length: 7 }).map((_, i) => ({
  date: format(subDays(new Date(), 6 - i), "MMM dd"),
  analyses: Math.floor(Math.random() * 12) + 3,
}));

const DR_GRADE_COLORS: Record<string, string> = {
  "0": "bg-green-500",
  "1": "bg-yellow-400",
  "2": "bg-orange-500",
  "3": "bg-red-500",
  "4": "bg-red-700",
};

const DR_GRADE_LABELS: Record<string, string> = {
  "0": "No DR",
  "1": "Mild",
  "2": "Moderate",
  "3": "Severe",
  "4": "Proliferative",
};

async function fetchRetinaStats() {
  const res = await fetch("/api/retina/cases/stats");
  if (!res.ok) throw new Error("Stats unavailable");
  return res.json();
}

async function fetchRecentCases() {
  const res = await fetch("/api/retina/cases?limit=5");
  if (!res.ok) throw new Error("Cases unavailable");
  return res.json();
}

export default function Dashboard() {
  const { data: patients, isLoading: pLoading } = useListPatients();
  const { data: stats } = useQuery({
    queryKey: ["retina-stats"],
    queryFn: fetchRetinaStats,
    staleTime: 30_000,
  });
  const { data: recentCasesData } = useQuery({
    queryKey: ["retina-recent-cases"],
    queryFn: fetchRecentCases,
    staleTime: 30_000,
  });

  const totalPatients = patients?.length || 0;
  const totalCases = stats?.total_cases ?? 0;
  const referrals = stats?.referrals_needed ?? 0;
  const gradeDistribution = stats?.grade_distribution || {};
  const recentCases = recentCasesData?.cases || [];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-display font-bold text-slate-900">Dashboard</h1>
          <p className="text-slate-500 mt-1">Welcome back. Here's what's happening today.</p>
        </div>
        <div className="flex gap-3">
          <Link href="/patients">
            <button className="px-4 py-2.5 rounded-xl font-semibold bg-white border border-slate-200 text-slate-700 shadow-sm hover:bg-slate-50 hover:border-slate-300 transition-all">
              View Patients
            </button>
          </Link>
          <Link href="/analyses/new">
            <button className="px-4 py-2.5 rounded-xl font-semibold bg-gradient-to-r from-primary to-blue-600 text-white shadow-md shadow-primary/20 hover:shadow-lg hover:shadow-primary/30 hover:-translate-y-0.5 transition-all">
              New Analysis
            </button>
          </Link>
        </div>
      </div>

      {/* Hero Banner */}
      <div className="relative rounded-3xl overflow-hidden bg-slate-900 h-64 shadow-xl border border-slate-800">
        <img
          src={`${import.meta.env.BASE_URL}images/dashboard-hero.png`}
          alt="Medical AI Dashboard"
          className="absolute inset-0 w-full h-full object-cover opacity-60 mix-blend-overlay"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-slate-900 via-slate-900/80 to-transparent" />
        <div className="absolute inset-0 p-8 flex flex-col justify-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/20 border border-primary/30 text-primary-foreground w-fit mb-4 backdrop-blur-md">
            <Activity className="h-4 w-4" />
            <span className="text-sm font-medium">Retina-GPT v2.0 — Active</span>
          </div>
          <h2 className="text-3xl font-display font-bold text-white max-w-lg leading-tight">
            AI-Powered Clinical Precision for Retinal Diagnostics
          </h2>
          <p className="text-slate-300 mt-2 max-w-md">
            Detect 13 retinal conditions in seconds with DR grading, AMD staging, glaucoma screening, and Grad-CAM explainability.
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm hover:shadow-md transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm font-medium text-slate-500">Total Patients</p>
              <h3 className="text-3xl font-display font-bold text-slate-900 mt-1">
                {pLoading ? "—" : totalPatients}
              </h3>
            </div>
            <div className="h-12 w-12 rounded-full bg-blue-50 flex items-center justify-center">
              <Users className="h-6 w-6 text-blue-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
            <span className="text-green-600 font-medium">Active</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm hover:shadow-md transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm font-medium text-slate-500">AI Analyses</p>
              <h3 className="text-3xl font-display font-bold text-slate-900 mt-1">{totalCases}</h3>
            </div>
            <div className="h-12 w-12 rounded-full bg-indigo-50 flex items-center justify-center">
              <ScanEye className="h-6 w-6 text-indigo-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <span className="text-slate-500">Retina-GPT cases</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm hover:shadow-md transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm font-medium text-slate-500">Referrals Needed</p>
              <h3 className={cn("text-3xl font-display font-bold mt-1", referrals > 0 ? "text-red-600" : "text-slate-900")}>
                {referrals}
              </h3>
            </div>
            <div className="h-12 w-12 rounded-full bg-red-50 flex items-center justify-center">
              <ShieldAlert className="h-6 w-6 text-red-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            {referrals > 0 ? (
              <span className="text-red-600 font-medium">Requires attention</span>
            ) : (
              <span className="text-green-600 font-medium flex items-center gap-1">
                <CheckCircle2 className="h-3 w-3" /> All clear
              </span>
            )}
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm hover:shadow-md transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm font-medium text-slate-500">This Week</p>
              <h3 className="text-3xl font-display font-bold text-slate-900 mt-1">
                {stats?.this_week ?? 0}
              </h3>
            </div>
            <div className="h-12 w-12 rounded-full bg-teal-50 flex items-center justify-center">
              <Clock className="h-6 w-6 text-teal-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <span className="text-slate-500">New scans this week</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Chart */}
        <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-display font-bold text-slate-900">Analyses Overview (Last 7 Days)</h3>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={MOCK_CHART_DATA} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} />
                <Tooltip
                  cursor={{ fill: "#f1f5f9" }}
                  contentStyle={{ borderRadius: "12px", border: "none", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)" }}
                />
                <Bar dataKey="analyses" radius={[6, 6, 0, 0]}>
                  {MOCK_CHART_DATA.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={index === MOCK_CHART_DATA.length - 1 ? "#0284c7" : "#bae6fd"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* DR Grade Distribution */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 flex flex-col">
          <h3 className="text-lg font-display font-bold text-slate-900 mb-5">DR Grade Distribution</h3>
          {Object.keys(gradeDistribution).length === 0 ? (
            <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">
              No cases yet — run your first analysis
            </div>
          ) : (
            <div className="space-y-3 flex-1">
              {Object.entries(gradeDistribution).map(([grade, count]) => {
                const total = Object.values(gradeDistribution).reduce((a: number, b: any) => a + b, 0);
                const pct = total > 0 ? Math.round(((count as number) / total) * 100) : 0;
                return (
                  <div key={grade}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-medium text-slate-700">
                        Grade {grade} — {DR_GRADE_LABELS[grade] || ""}
                      </span>
                      <span className="text-slate-500">{count as number} ({pct}%)</span>
                    </div>
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={cn("h-full rounded-full", DR_GRADE_COLORS[grade] || "bg-slate-400")}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Recent Retina Cases */}
      {recentCases.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <div className="flex justify-between items-center mb-5">
            <h3 className="text-lg font-display font-bold text-slate-900 flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />
              Recent AI Analyses
            </h3>
          </div>
          <div className="space-y-3">
            {recentCases.map((c: any) => (
              <Link key={c.id} href={`/retina-analyses/${c.id}`}>
                <div className="flex items-center gap-4 p-3 rounded-xl hover:bg-slate-50 cursor-pointer transition-colors group border border-transparent hover:border-slate-200">
                  <div className={cn(
                    "h-10 w-10 rounded-full flex items-center justify-center shrink-0 text-sm font-bold",
                    c.dr_refer ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
                  )}>
                    {c.dr_grade ?? "?"}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold text-slate-900 group-hover:text-primary transition-colors text-sm truncate">
                      {c.dr_label || `Grade ${c.dr_grade}`}
                    </p>
                    <p className="text-xs text-slate-500">
                      Patient: {c.patient_id} •{" "}
                      {c.created_at ? format(new Date(c.created_at), "MMM dd, yyyy") : "Unknown date"}
                    </p>
                  </div>
                  {c.dr_refer && (
                    <span className="text-xs font-bold text-red-700 bg-red-50 border border-red-200 px-2 py-0.5 rounded shrink-0">
                      REFER
                    </span>
                  )}
                  <ArrowRight className="h-4 w-4 text-slate-300 group-hover:text-primary transition-colors shrink-0" />
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Recent Patients */}
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
        <h3 className="text-lg font-display font-bold text-slate-900 mb-5 flex items-center gap-2">
          <Users className="h-5 w-5 text-primary" />
          Recent Patients
        </h3>
        <div className="space-y-3">
          {pLoading ? (
            <div className="animate-pulse space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-slate-100 rounded-full" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-slate-100 rounded w-1/2" />
                    <div className="h-3 bg-slate-100 rounded w-1/3" />
                  </div>
                </div>
              ))}
            </div>
          ) : patients?.length === 0 ? (
            <p className="text-sm text-slate-500 text-center py-4">No patients found.</p>
          ) : (
            patients?.slice(0, 5).map((patient) => (
              <Link key={patient.id} href={`/patients/${patient.id}`}>
                <div className="flex items-center gap-3 p-2 rounded-xl hover:bg-slate-50 cursor-pointer transition-colors group">
                  <div className="w-10 h-10 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center font-bold text-sm">
                    {patient.name.charAt(0)}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-slate-900 group-hover:text-primary transition-colors">
                      {patient.name}
                    </p>
                    <p className="text-xs text-slate-500">
                      {patient.age}y • {patient.gender}
                    </p>
                  </div>
                  <ArrowRight className="h-4 w-4 text-slate-300 group-hover:text-primary transition-colors" />
                </div>
              </Link>
            ))
          )}
        </div>
        <Link href="/patients">
          <button className="w-full mt-4 py-2 text-sm font-semibold text-primary hover:bg-primary/5 rounded-lg transition-colors">
            View All Patients
          </button>
        </Link>
      </div>
    </div>
  );
}
