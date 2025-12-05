import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface SimplePieChartProps {
  data: { name: string; value: number; color: string }[];
  loading?: boolean;
}

export function SimplePieChart({ data, loading }: SimplePieChartProps) {
  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center">
        <div className="animate-pulse w-48 h-48 bg-muted/50 rounded-full" />
      </div>
    );
  }
  
  return (
    <ResponsiveContainer width="100%" height={240}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={50}
          outerRadius={80}
          paddingAngle={4}
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px',
            fontSize: '12px',
          }}
        />
        <Legend 
          verticalAlign="bottom"
          height={36}
          formatter={(value) => (
            <span style={{ color: 'hsl(var(--muted-foreground))', fontSize: '12px' }}>
              {value}
            </span>
          )}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
