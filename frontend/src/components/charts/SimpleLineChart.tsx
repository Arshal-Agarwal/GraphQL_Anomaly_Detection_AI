import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface SimpleLineChartProps {
  data: { time: string; queries: number }[];
  loading?: boolean;
}

export function SimpleLineChart({ data, loading }: SimpleLineChartProps) {
  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center">
        <div className="animate-pulse w-full h-full bg-muted/50 rounded-lg" />
      </div>
    );
  }
  
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis 
          dataKey="time" 
          tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
          axisLine={{ stroke: 'hsl(var(--border))' }}
          tickLine={false}
        />
        <YAxis 
          tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px',
            fontSize: '12px',
          }}
          labelStyle={{ color: 'hsl(var(--foreground))' }}
        />
        <Line 
          type="monotone" 
          dataKey="queries" 
          stroke="hsl(var(--primary))" 
          strokeWidth={2}
          dot={{ fill: 'hsl(var(--primary))', strokeWidth: 0, r: 4 }}
          activeDot={{ r: 6, fill: 'hsl(var(--primary))' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
