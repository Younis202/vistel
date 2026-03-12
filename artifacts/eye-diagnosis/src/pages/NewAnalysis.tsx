import { useState, useCallback } from "react";
import { useLocation } from "wouter";
import { useListPatients, useCreateAnalysis, CreateAnalysisBodyEyeSide } from "@workspace/api-client-react";
import { useDropzone } from "react-dropzone";
import { useBase64Upload } from "@/hooks/use-file-upload";
import { Upload, Image as ImageIcon, AlertCircle, ScanEye, CheckCircle2, Cpu } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const PROCESSING_STEPS = [
  "Uploading Image to Secure Cloud",
  "Running Quality Assessment Model",
  "Detecting Retinal Lesions",
  "Classifying Disease Vectors",
  "Generating Final Diagnostic Report"
];

export default function NewAnalysis() {
  const [, setLocation] = useLocation();
  const searchParams = new URLSearchParams(window.location.search);
  const initialPatientId = searchParams.get('patientId') || "";

  const { data: patients, isLoading: pLoading } = useListPatients();
  const createAnalysis = useCreateAnalysis();
  const { convertToBase64 } = useBase64Upload();

  const [patientId, setPatientId] = useState(initialPatientId);
  const [eyeSide, setEyeSide] = useState<CreateAnalysisBodyEyeSide>("left");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Simulation state
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const selected = acceptedFiles[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [] },
    maxFiles: 1,
    multiple: false
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!patientId) return setError("Please select a patient.");
    if (!file) return setError("Please upload a fundus image.");

    try {
      setIsProcessing(true);
      setError(null);
      
      // Start fake progress simulation for UX
      const interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev < PROCESSING_STEPS.length - 1) return prev + 1;
          clearInterval(interval);
          return prev;
        });
      }, 2000);

      const base64 = await convertToBase64(file);
      
      // This is a real API call to the backend which triggers the AI integration
      const result = await createAnalysis.mutateAsync({
        data: {
          patientId: parseInt(patientId),
          eyeSide,
          imageName: file.name,
          imageBase64: base64
        }
      });

      clearInterval(interval);
      setCurrentStep(PROCESSING_STEPS.length);
      
      // Short delay so user sees "Done" before navigating
      setTimeout(() => {
        setLocation(`/analyses/${result.id}`);
      }, 800);

    } catch (err: any) {
      setIsProcessing(false);
      setError(err.message || "An error occurred during analysis.");
    }
  };

  if (isProcessing) {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center max-w-2xl mx-auto text-center px-4">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="relative w-32 h-32 mb-8"
        >
          <div className="absolute inset-0 rounded-full border-4 border-slate-100" />
          <motion.div 
            className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent border-r-transparent"
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
          />
          <div className="absolute inset-0 flex items-center justify-center">
            <Cpu className="h-10 w-10 text-primary" />
          </div>
        </motion.div>

        <h2 className="text-2xl font-display font-bold text-slate-900 mb-2">
          EyeWisdom AI is Analyzing
        </h2>
        <p className="text-slate-500 mb-12">Please wait while the multi-model pipeline processes the image.</p>

        <div className="w-full space-y-4">
          {PROCESSING_STEPS.map((step, idx) => {
            const isActive = currentStep === idx;
            const isDone = currentStep > idx;
            
            return (
              <motion.div 
                key={step}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className={cn(
                  "flex items-center gap-4 p-4 rounded-xl border transition-all",
                  isActive ? "bg-primary/5 border-primary/30 shadow-sm" : 
                  isDone ? "bg-green-50/50 border-green-100" : 
                  "bg-slate-50/50 border-transparent opacity-50"
                )}
              >
                <div className={cn(
                  "h-8 w-8 rounded-full flex items-center justify-center shrink-0",
                  isActive ? "bg-primary text-white" :
                  isDone ? "bg-green-500 text-white" :
                  "bg-slate-200 text-slate-400"
                )}>
                  {isDone ? <CheckCircle2 className="h-5 w-5" /> : <span className="text-sm font-bold">{idx + 1}</span>}
                </div>
                <span className={cn(
                  "font-medium text-left",
                  isActive ? "text-primary" :
                  isDone ? "text-green-700" :
                  "text-slate-500"
                )}>
                  {step}
                </span>
                {isActive && (
                  <motion.div 
                    className="ml-auto w-1.5 h-1.5 rounded-full bg-primary"
                    animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                    transition={{ repeat: Infinity, duration: 1 }}
                  />
                )}
              </motion.div>
            )
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-display font-bold text-slate-900">New AI Analysis</h1>
        <p className="text-slate-500 mt-1">Upload a fundus image for comprehensive diagnostic processing.</p>
      </div>

      <div className="bg-white rounded-3xl p-6 md:p-8 shadow-sm border border-slate-200">
        <form onSubmit={handleSubmit} className="space-y-8">
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-900">Patient</label>
              <select 
                value={patientId}
                onChange={e => setPatientId(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all text-slate-900"
                required
              >
                <option value="" disabled>Select a patient...</option>
                {patients?.map(p => (
                  <option key={p.id} value={p.id}>{p.name} ({p.age}y)</option>
                ))}
              </select>
              {patients?.length === 0 && !pLoading && (
                <p className="text-xs text-amber-600 mt-1">No patients available. Create one first.</p>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-900">Eye Side</label>
              <div className="grid grid-cols-3 gap-2">
                {(['left', 'right', 'both'] as const).map(side => (
                  <button
                    key={side}
                    type="button"
                    onClick={() => setEyeSide(side)}
                    className={cn(
                      "py-3 px-2 rounded-xl text-sm font-medium border capitalize transition-all",
                      eyeSide === side 
                        ? "bg-primary text-white border-primary shadow-md shadow-primary/20" 
                        : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"
                    )}
                  >
                    {side}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-900">Fundus Image</label>
            <div 
              {...getRootProps()} 
              className={cn(
                "border-2 border-dashed rounded-2xl p-8 transition-all cursor-pointer group flex flex-col items-center justify-center min-h-[240px]",
                isDragActive ? "border-primary bg-primary/5" : 
                file ? "border-slate-200 bg-slate-50" : "border-slate-300 hover:border-primary hover:bg-slate-50"
              )}
            >
              <input {...getInputProps()} />
              
              {preview ? (
                <div className="relative w-full h-full max-h-[300px] flex items-center justify-center">
                  <img src={preview} alt="Preview" className="max-h-full max-w-full rounded-lg shadow-sm object-contain" />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                    <p className="text-white font-medium flex items-center gap-2">
                      <Upload className="h-5 w-5" /> Change Image
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-center">
                  <div className="w-16 h-16 rounded-full bg-primary/10 text-primary flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                    <ImageIcon className="h-8 w-8" />
                  </div>
                  <p className="text-lg font-medium text-slate-900">Drag & drop image here</p>
                  <p className="text-sm text-slate-500 mt-1">or click to browse from your computer</p>
                  <p className="text-xs text-slate-400 mt-4">Supports JPEG, PNG (High resolution recommended)</p>
                </div>
              )}
            </div>
          </div>

          {error && (
            <div className="p-4 rounded-xl bg-red-50 border border-red-100 flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" />
              <p className="text-sm text-red-700 font-medium">{error}</p>
            </div>
          )}

          <div className="pt-4 border-t border-slate-100 flex justify-end">
            <button 
              type="submit"
              disabled={!file || !patientId}
              className="flex items-center gap-2 px-8 py-3.5 rounded-xl font-bold bg-gradient-to-r from-primary to-blue-600 text-white shadow-lg shadow-primary/25 hover:shadow-xl hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed transition-all text-lg"
            >
              <ScanEye className="h-6 w-6" />
              Run AI Diagnosis
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
