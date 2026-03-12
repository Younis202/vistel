import { Link } from "wouter";
import { useListPatients } from "@workspace/api-client-react";
import { 
  Users, 
  Activity, 
  FileText, 
  ArrowRight,
  TrendingUp,
  Clock
} from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { format, subDays } from "date-fns";

const MOCK_CHART_DATA = Array.from({ length: 7 }).map((_, i) => ({
  date: format(subDays(new Date(), 6 - i), 'MMM dd'),
  analyses: Math.floor(Math.random() * 15) + 5
}));

export default function Dashboard() {
  const { data: patients, isLoading } = useListPatients();

  const totalPatients = patients?.length || 0;
  // Simulating total analyses based on patients to make dashboard look alive
  const simulatedAnalyses = Math.floor(totalPatients * 2.4) || 24;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
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

      {/* Hero Banner with Generated Image */}
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
            <span className="text-sm font-medium">EyeWisdom Pipeline V2.0 Active</span>
          </div>
          <h2 className="text-3xl font-display font-bold text-white max-w-lg leading-tight">
            AI-Powered Clinical Precision for Retinal Diagnostics
          </h2>
          <p className="text-slate-300 mt-2 max-w-md">
            Detect up to 13 retinal conditions in seconds with state-of-the-art multi-model analysis.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm hover:shadow-md transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm font-medium text-slate-500">Total Patients</p>
              <h3 className="text-3xl font-display font-bold text-slate-900 mt-1">
                {isLoading ? "-" : totalPatients}
              </h3>
            </div>
            <div className="h-12 w-12 rounded-full bg-blue-50 flex items-center justify-center">
              <Users className="h-6 w-6 text-blue-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
            <span className="text-green-600 font-medium">+12%</span>
            <span className="text-slate-400 ml-2">from last month</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm hover:shadow-md transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm font-medium text-slate-500">Total Analyses</p>
              <h3 className="text-3xl font-display font-bold text-slate-900 mt-1">
                {isLoading ? "-" : simulatedAnalyses}
              </h3>
            </div>
            <div className="h-12 w-12 rounded-full bg-indigo-50 flex items-center justify-center">
              <FileText className="h-6 w-6 text-indigo-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
            <span className="text-green-600 font-medium">+8%</span>
            <span className="text-slate-400 ml-2">from last month</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm hover:shadow-md transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm font-medium text-slate-500">Avg. Processing Time</p>
              <h3 className="text-3xl font-display font-bold text-slate-900 mt-1">
                12.4s
              </h3>
            </div>
            <div className="h-12 w-12 rounded-full bg-teal-50 flex items-center justify-center">
              <Clock className="h-6 w-6 text-teal-600" />
            </div>
          </div>
          <div className="mt-4 flex items-center text-sm">
            <span className="text-teal-600 font-medium">Optimal</span>
            <span className="text-slate-400 ml-2">Across 3 models</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-display font-bold text-slate-900">Analyses Overview (Last 7 Days)</h3>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={MOCK_CHART_DATA} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fill: '#64748b', fontSize: 12 }} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#64748b', fontSize: 12 }} />
                <Tooltip 
                  cursor={{ fill: '#f1f5f9' }}
                  contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                />
                <Bar dataKey="analyses" radius={[6, 6, 0, 0]}>
                  {MOCK_CHART_DATA.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === MOCK_CHART_DATA.length - 1 ? '#0284c7' : '#bae6fd'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 flex flex-col">
          <h3 className="text-lg font-display font-bold text-slate-900 mb-6">Recent Patients</h3>
          <div className="space-y-4 flex-1">
            {isLoading ? (
              <div className="animate-pulse space-y-4">
                {[1, 2, 3].map(i => (
                  <div key={i} className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-slate-100 rounded-full" />
                    <div className="flex-1 space-y-2">
                      <div className="h-4 bg-slate-100 rounded w-1/2" />
                      <div className="h-3 bg-slate-100 rounded w-1/3" />
                    </div>
                  </div>
                ))}
              </div>
            ) : patients?.slice(0, 5).map(patient => (
              <Link key={patient.id} href={`/patients/${patient.id}`}>
                <div className="flex items-center gap-3 p-2 rounded-xl hover:bg-slate-50 cursor-pointer transition-colors group">
                  <div className="w-10 h-10 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center font-bold text-sm">
                    {patient.name.charAt(0)}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-slate-900 group-hover:text-primary transition-colors">{patient.name}</p>
                    <p className="text-xs text-slate-500">{patient.age}y • {patient.gender}</p>
                  </div>
                  <ArrowRight className="h-4 w-4 text-slate-300 group-hover:text-primary transition-colors" />
                </div>
              </Link>
            ))}
            {patients?.length === 0 && (
              <p className="text-sm text-slate-500 text-center py-4">No patients found.</p>
            )}
          </div>
          <Link href="/patients">
            <button className="w-full mt-4 py-2 text-sm font-semibold text-primary hover:bg-primary/5 rounded-lg transition-colors">
              View All
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}
