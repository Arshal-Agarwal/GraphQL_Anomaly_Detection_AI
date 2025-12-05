import { useCallback, useState } from 'react';
import { Upload, File, X, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

interface FileUploadBoxProps {
  onFileContent: (content: string) => void;
}

const ACCEPTED_TYPES = ['.graphql', '.gql', '.txt'];

export function FileUploadBox({ onFileContent }: FileUploadBoxProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  
  const handleFile = useCallback(async (file: File) => {
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!ACCEPTED_TYPES.includes(extension)) {
      toast.error('Invalid file type', {
        description: 'Please upload .graphql, .gql, or .txt files',
      });
      return;
    }
    
    try {
      const content = await file.text();
      setUploadedFile(file);
      onFileContent(content);
      toast.success('File uploaded successfully', {
        description: `${file.name} loaded`,
      });
    } catch (error) {
      toast.error('Failed to read file', {
        description: 'Please try again',
      });
    }
  }, [onFileContent]);
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);
  
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);
  
  const clearFile = useCallback(() => {
    setUploadedFile(null);
  }, []);
  
  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={cn(
        "relative rounded-xl border-2 border-dashed transition-all duration-200 p-6",
        isDragging 
          ? "border-primary bg-primary/5" 
          : "border-border hover:border-primary/50 hover:bg-muted/30",
        uploadedFile && "border-success bg-success/5"
      )}
    >
      {uploadedFile ? (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-success/10 flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-success" />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">{uploadedFile.name}</p>
              <p className="text-xs text-muted-foreground">
                {(uploadedFile.size / 1024).toFixed(1)} KB
              </p>
            </div>
          </div>
          <button
            onClick={clearFile}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
      ) : (
        <label className="flex flex-col items-center cursor-pointer">
          <div className={cn(
            "w-12 h-12 rounded-xl flex items-center justify-center mb-3 transition-colors",
            isDragging ? "bg-primary/20" : "bg-muted"
          )}>
            <Upload className={cn(
              "w-6 h-6 transition-colors",
              isDragging ? "text-primary" : "text-muted-foreground"
            )} />
          </div>
          <p className="text-sm font-medium text-foreground mb-1">
            Drop your file here or <span className="text-primary">browse</span>
          </p>
          <p className="text-xs text-muted-foreground">
            Supports .graphql, .gql, .txt
          </p>
          <input
            type="file"
            accept=".graphql,.gql,.txt"
            onChange={handleInputChange}
            className="hidden"
          />
        </label>
      )}
    </div>
  );
}
