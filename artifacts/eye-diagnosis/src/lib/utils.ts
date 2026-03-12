import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatRiskColor(risk: string) {
  switch (risk.toLowerCase()) {
    case 'normal': return 'bg-green-50 text-green-700 ring-green-600/20 border-green-200';
    case 'low': return 'bg-teal-50 text-teal-700 ring-teal-600/20 border-teal-200';
    case 'moderate': return 'bg-yellow-50 text-yellow-700 ring-yellow-600/20 border-yellow-200';
    case 'high': return 'bg-orange-50 text-orange-700 ring-orange-600/20 border-orange-200';
    case 'critical': return 'bg-red-50 text-red-700 ring-red-600/20 border-red-200';
    default: return 'bg-slate-50 text-slate-700 ring-slate-600/20 border-slate-200';
  }
}

export function formatSeverityColor(severity: string) {
  switch (severity.toLowerCase()) {
    case 'none': return 'bg-green-100 text-green-800';
    case 'mild': return 'bg-teal-100 text-teal-800';
    case 'moderate': return 'bg-yellow-100 text-yellow-800';
    case 'severe': return 'bg-red-100 text-red-800';
    default: return 'bg-slate-100 text-slate-800';
  }
}
