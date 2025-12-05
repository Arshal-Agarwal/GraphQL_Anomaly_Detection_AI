import { useEffect, useState } from 'react';
import { Cpu, HardDrive, Timer, Activity, AlertTriangle, RefreshCw } from 'lucide-react';
import { MetricsCard } from '@/components/MetricsCard';
import { SimpleLineChart } from '@/components/charts/SimpleLineChart';
import { SimplePieChart } from '@/components/charts/SimplePieChart';
import { Button } from '@/components/ui/button';
import { getMetrics, type SystemMetrics } from '@/lib/api';

const Dashboard = () => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  
  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchMetrics();
  }, []);
  
  return (
    <div className="min-h-screen bg-background pt-20 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-foreground">
              System Dashboard
            </h1>
            <p className="text-muted-foreground mt-1">
              Monitor detection performance and system health
            </p>
          </div>
          <Button
            variant="outline"
            onClick={fetchMetrics}
            disabled={loading}
            className="gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
        
        {/* Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          <MetricsCard
            title="Memory Usage"
            value={metrics ? `${metrics.memoryUsage.toFixed(0)}` : '0'}
            suffix="%"
            icon={HardDrive}
            change={-5}
            loading={loading}
          />
          <MetricsCard
            title="CPU Load"
            value={metrics ? `${metrics.cpuLoad.toFixed(0)}` : '0'}
            suffix="%"
            icon={Cpu}
            change={12}
            loading={loading}
          />
          <MetricsCard
            title="Avg Latency"
            value={metrics ? `${metrics.avgLatency.toFixed(0)}` : '0'}
            suffix="ms"
            icon={Timer}
            change={-8}
            loading={loading}
          />
          <MetricsCard
            title="Total Queries"
            value={metrics?.totalQueries.toLocaleString() || '0'}
            icon={Activity}
            change={23}
            loading={loading}
          />
          <MetricsCard
            title="Anomalies Today"
            value={metrics?.anomaliesToday || '0'}
            icon={AlertTriangle}
            change={-15}
            loading={loading}
          />
        </div>
        
        {/* Charts */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Query Load Chart */}
          <div className="glass-card-elevated rounded-2xl p-6">
            <h3 className="text-lg font-semibold text-foreground mb-6">Query Load Over Time</h3>
            <SimpleLineChart
              data={metrics?.queryLoadOverTime || []}
              loading={loading}
            />
          </div>
          
          {/* Anomaly Categories */}
          <div className="glass-card-elevated rounded-2xl p-6">
            <h3 className="text-lg font-semibold text-foreground mb-6">Anomaly Categories</h3>
            <SimplePieChart
              data={metrics?.anomalyCategories || []}
              loading={loading}
            />
          </div>
        </div>
        
        {/* Recent Activity */}
        <div className="glass-card-elevated rounded-2xl p-6 mt-6">
          <h3 className="text-lg font-semibold text-foreground mb-4">Recent Detections</h3>
          <div className="space-y-3">
            {[
              { time: '2 min ago', type: 'suspicious', query: 'Introspection query detected' },
              { time: '15 min ago', type: 'normal', query: 'Standard user query' },
              { time: '32 min ago', type: 'dangerous', query: 'Deep nesting attack attempt' },
              { time: '1 hour ago', type: 'suspicious', query: 'Batch query abuse detected' },
            ].map((item, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-4 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${
                    item.type === 'normal' ? 'bg-success' :
                    item.type === 'suspicious' ? 'bg-warning' : 'bg-danger'
                  }`} />
                  <span className="text-sm text-foreground">{item.query}</span>
                </div>
                <span className="text-xs text-muted-foreground">{item.time}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
