import { useParams, Link } from "wouter";
import { useGetPatient, useListPatientAnalyses } from "@workspace/api-client-react";
import { ArrowLeft, UserCircle, Activity, Calendar, ScanEye, ChevronRight } from "lucide-react";
import { format } from "date-fns";
import { formatRiskColor, cn } from "@/lib/utils";

export default function PatientDetail() {
  const { id } = useParams<{ id: string }>();
  const patientId = parseInt(id || "0");

  const { data: patient, isLoading: pLoading } = useGetPatient(patientId);
  const { data: analyses, isLoading: aLoading } = useListPatientAnalyses(patientId);

  if (pLoading) {
    return <div className="p-8 text-center text-slate-500 animate-pulse">Loading patient profile...</div>;
  }

  if (!patient) {
    return <div className="p-8 text-center text-red-500">Patient not found</div>;
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <Link href="/patients">
        <button className="flex items-center text-sm font-medium text-slate-500 hover:text-slate-900 transition-colors mb-4">
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back to Patients
        </button>
      </Link>

      <div className="bg-white rounded-3xl p-8 border border-slate-100 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="flex items-center gap-6">
          <div className="h-24 w-24 rounded-2xl bg-gradient-to-br from-primary/20 to-blue-100 text-primary flex items-center justify-center shadow-inner">
            <UserCircle className="h-12 w-12" />
          </div>
          <div>
            <h1 className="text-3xl font-display font-bold text-slate-900">{patient.name}</h1>
            <div className="flex flex-wrap items-center gap-4 mt-2 text-slate-600">
              <span className="flex items-center gap-1">
                <span className="font-semibold text-slate-900">{patient.age}</span> yrs
              </span>
              <span className="w-1.5 h-1.5 rounded-full bg-slate-300" />
              <span className="capitalize">{patient.gender}</span>
              <span className="w-1.5 h-1.5 rounded-full bg-slate-300" />
              <span className="flex items-center gap-1">
                <Calendar className="h-4 w-4 text-slate-400" />
                Added {format(new Date(patient.createdAt), 'MMM yyyy')}
              </span>
            </div>
          </div>
        </div>
        <Link href={`/analyses/new?patientId=${patient.id}`}>
          <button className="flex items-center gap-2 px-5 py-3 rounded-xl font-semibold bg-gradient-to-r from-primary to-blue-600 text-white shadow-md shadow-primary/20 hover:shadow-lg hover:-translate-y-0.5 transition-all">
            <ScanEye className="h-5 w-5" />
            New Analysis
          </button>
        </Link>
      </div>

      <div>
        <h2 className="text-xl font-display font-bold text-slate-900 mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary" />
          Diagnostic History
        </h2>

        {aLoading ? (
          <div className="animate-pulse space-y-4">
            {[1, 2].map(i => <div key={i} className="h-32 bg-white rounded-2xl border border-slate-100" />)}
          </div>
        ) : analyses?.length === 0 ? (
          <div className="bg-slate-50 border border-dashed border-slate-200 rounded-2xl p-12 text-center">
            <ScanEye className="h-12 w-12 text-slate-300 mx-auto mb-3" />
            <h3 className="text-lg font-medium text-slate-900 mb-1">No analyses yet</h3>
            <p className="text-slate-500 mb-4">Run an AI analysis on a fundus image to get started.</p>
            <Link href={`/analyses/new?patientId=${patient.id}`}>
              <button className="px-4 py-2 rounded-lg font-medium text-primary bg-primary/10 hover:bg-primary/20 transition-colors">
                Run First Analysis
              </button>
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {analyses?.map(analysis => {
              const detectedCount = analysis.diseases.filter(d => d.detected).length;
              return (
                <Link key={analysis.id} href={`/analyses/${analysis.id}`}>
                  <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm hover:shadow-md hover:border-primary/20 transition-all cursor-pointer group flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div className="flex items-center gap-6">
                      <div className="w-16 h-16 rounded-xl overflow-hidden bg-slate-100 border border-slate-200 shrink-0">
                        {analysis.imageUrl ? (
                          <img src={analysis.imageUrl} alt="Fundus" className="w-full h-full object-cover" />
                        ) : (
                          <ScanEye className="w-8 h-8 text-slate-300 m-auto mt-4" />
                        )}
                      </div>
                      <div>
                        <div className="flex items-center gap-3 mb-1">
                          <h3 className="font-semibold text-lg text-slate-900">
                            {format(new Date(analysis.createdAt), 'MMM dd, yyyy')}
                          </h3>
                          <span className="px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-600 capitalize border border-slate-200">
                            {analysis.eyeSide} Eye
                          </span>
                        </div>
                        <p className="text-sm text-slate-500">
                          {detectedCount === 0 
                            ? "No notable findings." 
                            : `${detectedCount} condition${detectedCount > 1 ? 's' : ''} detected.`}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-6 self-start md:self-auto">
                      <div className="text-right">
                        <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold mb-1">Overall Risk</p>
                        <div className={cn("px-3 py-1 rounded-full text-sm font-semibold border inline-block capitalize", formatRiskColor(analysis.overallRisk))}>
                          {analysis.overallRisk}
                        </div>
                      </div>
                      <div className="h-10 w-10 rounded-full bg-slate-50 flex items-center justify-center group-hover:bg-primary group-hover:text-white text-slate-400 transition-colors">
                        <ChevronRight className="h-5 w-5" />
                      </div>
                    </div>
                  </div>
                </Link>
              )
            })}
          </div>
        )}
      </div>
    </div>
  );
}
