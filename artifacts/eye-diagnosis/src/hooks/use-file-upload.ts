import { useState, useCallback } from 'react';

export function useBase64Upload() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const convertToBase64 = useCallback(async (file: File): Promise<string> => {
    setIsProcessing(true);
    setError(null);
    
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      
      reader.onload = () => {
        setIsProcessing(false);
        resolve(reader.result as string);
      };
      
      reader.onerror = (err) => {
        setIsProcessing(false);
        setError('Failed to read file');
        reject(err);
      };
    });
  }, []);

  return { convertToBase64, isProcessing, error };
}
