import { cn } from '@/lib/utils';
import { LucideIcon, TrendingUp, TrendingDown } from 'lucide-react';

interface MetricsCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  change?: number;
  suffix?: string;
  loading?: boolean;
}

export function MetricsCard({ 
  title, 
  value, 
  icon: Icon, 
  change, 
  suffix,
  loading 
}: MetricsCardProps) {
  const isPositive = change && change > 0;
  
  return (
    <div className="metric-card">
      {loading ? (
        <div className="animate-pulse">
          <div className="h-4 w-24 bg-muted rounded mb-3" />
          <div className="h-8 w-16 bg-muted rounded mb-2" />
          <div className="h-3 w-20 bg-muted rounded" />
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-muted-foreground">{title}</span>
            <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
              <Icon className="w-5 h-5 text-primary" />
            </div>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-foreground">{value}</span>
            {suffix && <span className="text-sm text-muted-foreground">{suffix}</span>}
          </div>
          {change !== undefined && (
            <div className={cn(
              "flex items-center gap-1 mt-2 text-xs font-medium",
              isPositive ? "text-success" : "text-danger"
            )}>
              {isPositive ? (
                <TrendingUp className="w-3 h-3" />
              ) : (
                <TrendingDown className="w-3 h-3" />
              )}
              <span>{Math.abs(change)}% from yesterday</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
