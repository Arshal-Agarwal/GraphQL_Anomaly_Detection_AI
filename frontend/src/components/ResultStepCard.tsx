import { cn } from '@/lib/utils';
import { CheckCircle, AlertTriangle, XCircle, Info, ChevronRight } from 'lucide-react';
import type { AnalysisResult, StructuralFeature } from '@/lib/api';

interface ResultStepCardProps {
  result: AnalysisResult;
}

const statusConfig = {
  normal: {
    icon: CheckCircle,
    label: 'Normal',
    className: 'status-normal',
    bgClass: 'bg-success/10',
    textClass: 'text-success',
  },
  suspicious: {
    icon: AlertTriangle,
    label: 'Suspicious',
    className: 'status-suspicious',
    bgClass: 'bg-warning/10',
    textClass: 'text-warning',
  },
  dangerous: {
    icon: XCircle,
    label: 'Dangerous',
    className: 'status-dangerous',
    bgClass: 'bg-danger/10',
    textClass: 'text-danger',
  },
};

function FeatureRow({ feature }: { feature: StructuralFeature }) {
  return (
    <div className={cn(
      "flex items-center justify-between py-2 px-3 rounded-lg",
      feature.isAnomalous ? "bg-danger/5" : "bg-muted/50"
    )}>
      <span className="text-sm text-muted-foreground">{feature.name}</span>
      <span className={cn(
        "text-sm font-medium",
        feature.isAnomalous ? "text-danger" : "text-foreground"
      )}>
        {feature.value}
        {feature.isAnomalous && (
          <AlertTriangle className="inline-block w-3 h-3 ml-1.5" />
        )}
      </span>
    </div>
  );
}

export function ResultStepCard({ result }: ResultStepCardProps) {
  const status = statusConfig[result.anomalyFlag];
  const StatusIcon = status.icon;
  
  return (
    <div className="space-y-4">
      {/* Step 1: Summary */}
      <div className="glass-card rounded-xl p-5 animate-slide-up opacity-0 stagger-1">
        <div className="flex items-start gap-3 mb-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
            <span className="text-sm font-semibold text-primary">1</span>
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Query Summary</h3>
            <p className="text-sm text-muted-foreground mt-1">{result.summary}</p>
          </div>
        </div>
      </div>
      
      {/* Step 2: Structural Features */}
      <div className="glass-card rounded-xl p-5 animate-slide-up opacity-0 stagger-2">
        <div className="flex items-start gap-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
            <span className="text-sm font-semibold text-primary">2</span>
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-foreground">Structural Features</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Detected {result.features.length} features
            </p>
          </div>
        </div>
        <div className="space-y-2 ml-11">
          {result.features.map((feature, idx) => (
            <FeatureRow key={idx} feature={feature} />
          ))}
        </div>
      </div>
      
      {/* Step 3: Anomaly Detection */}
      <div className="glass-card rounded-xl p-5 animate-slide-up opacity-0 stagger-3">
        <div className="flex items-start gap-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
            <span className="text-sm font-semibold text-primary">3</span>
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-foreground">Anomaly Detection</h3>
          </div>
        </div>
        <div className="ml-11 flex items-center gap-4">
          <div className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg border",
            status.className
          )}>
            <StatusIcon className="w-5 h-5" />
            <span className="font-semibold">{status.label}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Confidence:</span>
            <span className="text-lg font-bold text-foreground">{result.confidenceScore}%</span>
          </div>
        </div>
      </div>
      
      {/* Step 4: Remediation */}
      <div className="glass-card rounded-xl p-5 animate-slide-up opacity-0 stagger-4">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
            <span className="text-sm font-semibold text-primary">4</span>
          </div>
          <div>
            <h3 className="font-semibold text-foreground mb-2">Recommendation</h3>
            <div className={cn(
              "flex items-start gap-3 p-4 rounded-lg",
              status.bgClass
            )}>
              <Info className={cn("w-5 h-5 shrink-0 mt-0.5", status.textClass)} />
              <p className={cn("text-sm", status.textClass)}>{result.remediation}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
