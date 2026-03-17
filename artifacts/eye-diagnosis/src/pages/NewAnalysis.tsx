import { useState, useCallback } from "react";
import { useLocation } from "wouter";
import { useListPatients } from "@workspace/api-client-react";
import { useDropzone } from "react-dropzone";
import { Upload, Image as ImageIcon, AlertCircle, ScanEye, CheckCircle2, Cpu, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

const PROCESSING_STEPS = [
  "Uploading Image to Secure Server",
  "Running Image Quality Assessment",
  "Detecting Retinal Lesions (8 categories)",
  "Grading Diabetic Retinopathy (0–4)",
  "Screening for AMD & Glaucoma",
  "Generating Clinical Report",
  "Finalising AI Diagnostic Summary",
];

export default function NewAnalysis() {
  const [, setLocation] = useLocation();
  const searchParams = new URLSearchParams(window.location.search);
  const initialPatientId = searchParams.get("patientId") || "";

  const { data: patients, isLoading: pLoading } = useListPatients();

  const [patientId, setPatientId] = useState(initialPatientId);
  const [eyeSide, setEyeSide] = useState<"left" | "right">("left");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [done, setDone] = useState(false);

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
    accept: { "image/jpeg": [], "image/png": [], "image/tiff": [], "image/bmp": [] },
    maxFiles: 1,
    multiple: false,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return setError("Please upload a fundus image.");

    try {
      setIsProcessing(true);
      setCurrentStep(0);
      setError(null);
      setDone(false);

      const stepInterval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev < PROCESSING_STEPS.length - 2) return prev + 1;
          return prev;
        });
      }, 900);

      const formData = new FormData();
      formData.append("file", file, file.name);
      formData.append("explain", "true");
      formData.append("segment", "false");
      if (patientId) formData.append("patient_id", patientId);

      const res = await fetch("/api/retina/analyze", {
        method: "POST",
        body: formData,
      });

      clearInterval(stepInterval);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Analysis failed" }));
        throw new Error(err.error || err.detail || "Analysis failed");
      }

      const result = await res.json();
      setCurrentStep(PROCESSING_STEPS.length - 1);
      setDone(true);

      await new Promise((r) => setTimeout(r, 800));
      setLocation(`/retina-analyses/${result.image_id}`);
    } catch (err: any) {
      setError(err.message || "Analysis failed. Please try again.");
      setIsProcessing(false);
      setCurrentStep(0);
    }
  };

  if (isProcessing) {
    return (
      <div className="max-w-2xl mx-auto py-20 px-4">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-3xl shadow-lg border border-slate-200 p-10 text-center space-y-8"
        >
          <div className="flex justify-center">
            <div className="relative">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                className="h-20 w-20 rounded-full border-4 border-primary/20 border-t-primary"
              />
              <ScanEye className="absolute inset-0 m-auto h-8 w-8 text-primary" />
            </div>
          </div>

          <div>
            <h2 className="text-2xl font-display font-bold text-slate-900 mb-2">
              {done ? "Analysis Complete" : "AI Analysis in Progress"}
            </h2>
            <p className="text-slate-500 text-sm">
              {done ? "Redirecting to your report..." : "Retina-GPT is examining your fundus image"}
            </p>
          </div>

          <div className="space-y-2">
            {PROCESSING_STEPS.map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
                className={cn(
                  "flex items-center gap-3 py-2.5 px-4 rounded-xl text-sm transition-all",
                  i < currentStep
                    ? "bg-green-50 text-green-800"
                    : i === currentStep
                    ? "bg-primary/10 text-primary font-semibold"
                    : "text-slate-400"
                )}
              >
                {i < currentStep ? (
                  <CheckCircle2 className="h-4 w-4 text-green-600 shrink-0" />
                ) : i === currentStep ? (
                  <Loader2 className="h-4 w-4 text-primary shrink-0 animate-spin" />
                ) : (
                  <div className="h-4 w-4 rounded-full border border-slate-300 shrink-0" />
                )}
                {step}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 pb-12">
      <div>
        <h1 className="text-3xl font-display font-bold text-slate-900">New Retinal Analysis</h1>
        <p className="text-slate-500 mt-1">Upload a fundus image for AI-powered diagnosis across 13 pathologies</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Patient Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Patient</label>
              <select
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                className="w-full border border-slate-200 rounded-xl px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-primary/30"
                disabled={pLoading}
              >
                <option value="">Select patient (optional)</option>
                {patients?.map((p: any) => (
                  <option key={p.id} value={String(p.id)}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Eye Side</label>
              <div className="flex gap-2">
                {(["left", "right"] as const).map((side) => (
                  <button
                    key={side}
                    type="button"
                    onClick={() => setEyeSide(side)}
                    className={cn(
                      "flex-1 py-2 rounded-xl text-sm font-medium border capitalize transition-all",
                      eyeSide === side
                        ? "bg-primary text-white border-primary shadow-sm"
                        : "bg-white text-slate-600 border-slate-200 hover:border-primary/40"
                    )}
                  >
                    {side} eye
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Fundus Image Upload</h2>

          <div
            {...getRootProps()}
            className={cn(
              "border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all",
              isDragActive
                ? "border-primary bg-primary/5"
                : preview
                ? "border-green-400 bg-green-50"
                : "border-slate-200 hover:border-primary/50 hover:bg-slate-50"
            )}
          >
            <input {...getInputProps()} />
            {preview ? (
              <div className="space-y-3">
                <img
                  src={preview}
                  alt="Preview"
                  className="h-48 mx-auto rounded-xl object-contain border border-slate-200 shadow-sm"
                />
                <p className="text-sm font-medium text-slate-600">{file?.name}</p>
                <p className="text-xs text-slate-400">Click or drag to replace</p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="mx-auto h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center">
                  {isDragActive ? (
                    <Upload className="h-8 w-8 text-primary" />
                  ) : (
                    <ImageIcon className="h-8 w-8 text-primary" />
                  )}
                </div>
                <div>
                  <p className="font-semibold text-slate-800">
                    {isDragActive ? "Drop image here" : "Drag & drop fundus image"}
                  </p>
                  <p className="text-sm text-slate-500 mt-1">or click to browse — JPEG, PNG, TIFF supported</p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-3xl p-6 text-white border border-slate-700">
          <div className="flex items-center gap-3 mb-4">
            <Cpu className="h-5 w-5 text-primary" />
            <h2 className="font-semibold">Retina-GPT AI Capabilities</h2>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
            {[
              "DR Grading (0–4)",
              "AMD Staging (0–3)",
              "Glaucoma Detection",
              "Microaneurysm Detection",
              "Hemorrhage Detection",
              "Hard Exudate Detection",
              "Soft Exudate / CWS",
              "Neovascularization",
              "Drusen Detection",
              "Grad-CAM Explainability",
              "Image Quality Assessment",
              "Clinical Report Generation",
            ].map((feat) => (
              <div key={feat} className="flex items-center gap-1.5 text-slate-300">
                <CheckCircle2 className="h-3 w-3 text-green-400 shrink-0" />
                {feat}
              </div>
            ))}
          </div>
        </div>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="flex items-center gap-3 bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-xl"
            >
              <AlertCircle className="h-5 w-5 shrink-0" />
              <span className="text-sm">{error}</span>
            </motion.div>
          )}
        </AnimatePresence>

        <button
          type="submit"
          disabled={!file || isProcessing}
          className="w-full py-4 px-6 bg-primary text-white font-bold text-lg rounded-2xl shadow-lg hover:bg-primary/90 transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-3"
        >
          <ScanEye className="h-5 w-5" />
          Run AI Retinal Diagnosis
        </button>
      </form>
    </div>
  );
}
