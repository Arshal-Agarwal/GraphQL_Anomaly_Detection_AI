import { useState } from 'react';
import { Play, Loader2, Sparkles } from 'lucide-react';
import { QueryEditor } from '@/components/QueryEditor';
import { FileUploadBox } from '@/components/FileUploadBox';
import { ResultStepCard } from '@/components/ResultStepCard';
import { Button } from '@/components/ui/button';
import { analyzeQuery, type AnalysisResult } from '@/lib/api';
import { toast } from 'sonner';

const Index = () => {
  const [query, setQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  
  const handleAnalyze = async () => {
    if (!query.trim()) {
      toast.error('Please enter a GraphQL query');
      return;
    }
    
    setIsAnalyzing(true);
    setResult(null);
    
    try {
      const analysisResult = await analyzeQuery(query);
      setResult(analysisResult);
    } catch (error) {
      toast.error('Analysis failed', {
        description: 'Please try again',
      });
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  const handleFileContent = (content: string) => {
    setQuery(content);
    setResult(null);
  };
  
  return (
    <div className="min-h-screen bg-background pt-20 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-10 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4">
            <Sparkles className="w-4 h-4" />
            AI-Powered Analysis
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-foreground mb-3">
            GraphQL Query Analyzer
          </h1>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Detect anomalies, security issues, and performance problems in your GraphQL queries with ML-powered analysis.
          </p>
        </div>
        
        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left Panel - Input */}
          <div className="space-y-5">
            <div className="glass-card-elevated rounded-2xl p-5">
              <h2 className="text-lg font-semibold text-foreground mb-4">Query Input</h2>
              <QueryEditor value={query} onChange={setQuery} height="320px" />
              
              <div className="mt-4">
                <FileUploadBox onFileContent={handleFileContent} />
              </div>
              
              <Button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="w-full mt-5 h-12 text-base font-semibold"
                size="lg"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 mr-2" />
                    Analyze Query
                  </>
                )}
              </Button>
            </div>
          </div>
          
          {/* Right Panel - Results */}
          <div className="space-y-5">
            <div className="glass-card-elevated rounded-2xl p-5 min-h-[500px]">
              <h2 className="text-lg font-semibold text-foreground mb-4">Analysis Results</h2>
              
              {isAnalyzing ? (
                <div className="flex flex-col items-center justify-center h-80 gap-4">
                  <div className="relative">
                    <div className="w-16 h-16 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Analyzing query structure...
                  </p>
                </div>
              ) : result ? (
                <ResultStepCard result={result} />
              ) : (
                <div className="flex flex-col items-center justify-center h-80 text-center">
                  <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mb-4">
                    <Sparkles className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <p className="text-muted-foreground">
                    Enter a GraphQL query and click "Analyze" to see results
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
