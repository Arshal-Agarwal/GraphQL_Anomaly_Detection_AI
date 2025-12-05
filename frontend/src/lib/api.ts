// lib/api.ts

// --------------------------
// Types
// --------------------------

export interface StructuralFeature {
  name: string;
  value: number | string;
  isAnomalous: boolean;
}

export interface AnalysisResult {
  summary: string;
  features: StructuralFeature[];
  anomalyFlag: "normal" | "suspicious" | "dangerous";
  confidenceScore: number;
  remediation: string;
}

export interface SystemMetrics {
  memoryUsage: number;
  cpuLoad: number;
  avgLatency: number;
  totalQueries: number;
  anomaliesToday: number;
  queryLoadOverTime: { time: string; value: number }[];
  anomalyCategories: { label: string; value: number }[];
}


export async function analyzeQuery(query: string): Promise<AnalysisResult> {

  await new Promise((res) => setTimeout(res, 1000));


  try {

    fetch("http://localhost:3000/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    }).catch(() => {});
  } catch (_) {

  }


  const depth = computeDepth(query);
  const fieldCount = computeFieldCount(query);
  const cost = computeEstimatedCost(depth, fieldCount);
  const warnings = computeWarnings(query, depth, fieldCount, cost);

  const hasPagination = checkPagination(query);

  const features: StructuralFeature[] = [
    { name: "Depth", value: depth, isAnomalous: depth > 6 },
    { name: "Field Count", value: fieldCount, isAnomalous: fieldCount > 50 },
    { name: "Estimated Cost", value: cost, isAnomalous: cost > 250 },
    { name: "Has Introspection", value: query.includes("__schema") ? "yes" : "no", isAnomalous: query.includes("__schema") },

    // Pagination now a SOFT feature
    { 
      name: "Pagination Used",
      value: hasPagination ? "yes" : "no",
      isAnomalous: false
    },
  ];

  return {
    summary: `Depth=${depth}, fields=${fieldCount}, cost=${cost}.`,
    features,
    anomalyFlag: classifyAnomaly(depth, fieldCount, cost, warnings),
    confidenceScore: 85 + Math.floor(Math.random() * 10),
    remediation: warnings.length ? warnings.join(" ") : "No major risks detected."
  };
}

// --------------------------
// HELPERS
// --------------------------

function computeDepth(query: string): number {
  let depth = 0;
  let maxDepth = 0;

  for (const char of query) {
    if (char === "{") {
      depth++;
      maxDepth = Math.max(maxDepth, depth);
    } else if (char === "}") {
      depth--;
    }
  }
  return maxDepth;
}

function computeFieldCount(query: string): number {
  return query
    .split("\n")
    .map((l) => l.trim())
    .filter((l) =>
      /^[A-Za-z_][A-Za-z0-9_]*\s*(\(|{|$)/.test(l) &&
      !["query", "mutation", "subscription"].some((kw) => l.startsWith(kw))
    ).length;
}

function computeEstimatedCost(depth: number, fields: number): number {
  return depth * 10 + fields * 3 + Math.floor(depth * fields * 0.15);
}

function computeWarnings(query: string, depth: number, fields: number, cost: number) {
  const w: string[] = [];

  if (query.includes("__schema")) w.push("⚠️ Introspection detected.");
  if (depth > 6) w.push("⚠️ Excessive depth may cause heavy resolver chaining.");
  if (fields > 50) w.push("⚠️ Large number of fields — may be overfetching.");
  if (cost > 250) w.push("⚠️ High estimated computational cost.");

  // Pagination now soft
  if (!checkPagination(query)) {
    w.push("ℹ️ No pagination detected — may be fine, but consider limiting large lists.");
  }

  return w;
}

function classifyAnomaly(depth: number, fields: number, cost: number, warnings: string[]) {
  if (cost > 350 || depth > 10) return "dangerous";
  if (warnings.some((w) => w.includes("⚠️"))) return "suspicious";
  return "normal";
}

function checkPagination(query: string) {
  return /(first|last|limit)\s*:\s*\d+/i.test(query);
}

// --------------------------
// DASHBOARD METRICS
// --------------------------

export async function getMetrics(): Promise<SystemMetrics> {
  await new Promise((r) => setTimeout(r, 300));

  return {
    memoryUsage: 45.3,
    cpuLoad: 62.1,
    avgLatency: 124.7,
    totalQueries: 18523,
    anomaliesToday: 12,
    queryLoadOverTime: Array.from({ length: 10 }).map((_, i) => ({
      time: `${i * 10}m`,
      value: Math.floor(Math.random() * 100),
    })),
    anomalyCategories: [
      { label: "Dangerous", value: 12 },
      { label: "Suspicious", value: 34 },
      { label: "Normal", value: 54 },
    ],
  };
}
