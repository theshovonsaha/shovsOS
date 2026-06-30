export type SourceContract = {
  objective: string;
  entityCount: number;
  urlsPerEntity: number;
  totalUrls: number;
  requiredTools: string[];
  missing: string[];
};

export type TraceEvent = {
  kind?: string;
  tool?: string;
  entity?: string;
  url?: string;
  ok?: boolean;
  summary?: string;
};

export type TraceEval = {
  ok: boolean;
  score: number;
  failures: string[];
  metrics: {
    searchCount: number;
    fetchCount: number;
    requiredFetchCount: number;
    lockedEntities: string[];
  };
};

export type KernelDecision = {
  state: "act" | "respond";
  reason: string;
  nextTool?: string;
  nextArgs?: Record<string, string>;
};

const WORDS: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
};
const NUMBER_PATTERN = "\\d+|one|two|three|four|five|six|seven|eight|nine|ten";

function numberValue(raw: string | undefined): number {
  if (!raw) return 0;
  return /^\d+$/.test(raw) ? Number(raw) : WORDS[raw] ?? 0;
}

export function inferSourceContract(objective: string): SourceContract {
  const text = objective.toLowerCase().replace(/\s+/g, " ").trim();
  const entityCount = numberValue(text.match(new RegExp(`\\btop\\s+(${NUMBER_PATTERN})\\b`))?.[1]);
  const urlsPerEntity = numberValue(
    text.match(
      new RegExp(
        `\\b(${NUMBER_PATTERN})\\s+(?:relevant\\s+)?(?:urls?|links?|results?|sources?|articles?)\\s+(?:for\\s+)?(?:each|per)\\b`,
      ),
    )?.[1],
  );
  const requiredTools: string[] = [];
  if (/(search|find|lookup)/.test(text)) requiredTools.push("web_search");
  if (/(fetch|open|read)/.test(text)) requiredTools.push("web_fetch");
  if (requiredTools.includes("web_fetch") && !requiredTools.includes("web_search")) {
    requiredTools.unshift("web_search");
  }
  const missing: string[] = [];
  if (text.includes("each") && !entityCount) missing.push("entity_count");
  if (text.includes("each") && !urlsPerEntity) missing.push("urls_per_entity");
  return {
    objective,
    entityCount,
    urlsPerEntity,
    totalUrls: entityCount && urlsPerEntity ? entityCount * urlsPerEntity : 0,
    requiredTools,
    missing,
  };
}

export function discoveryQuery(objective: string): string {
  const stripped = objective
    .replace(
      /\b(?:then|and then|after that|web\s*search|search\s+each|search\s+those|search\s+these|web\s*fetch|fetch|capture|analy[sz]e|write|produce|report|tldr|tl;dr|summary table|one by one|separately|each)\b.*$/i,
      "",
    )
    .replace(/^\s*(?:please\s+)?(?:web\s*search|search\s+for|search|find|look\s*up)\s+/i, "")
    .replace(new RegExp(`\\b(top\\s+(?:${NUMBER_PATTERN})|those|these|(?:${NUMBER_PATTERN})\\s+relevant|all\\s+(?:${NUMBER_PATTERN})\\s+urls?)\\b`, "gi"), "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[.,:;]+$/g, "");
  return stripped || objective.slice(0, 80).trim();
}

export function evaluateTrace(contract: SourceContract, events: TraceEvent[]): TraceEval {
  const searches = events.filter((event) => event.tool === "web_search" && event.ok !== false);
  const fetches = events.filter((event) => event.tool === "web_fetch" && event.ok !== false);
  const lockedEntities = Array.from(
    new Set(
      events
        .filter((event) => event.kind === "entity_locked" && event.entity)
        .map((event) => String(event.entity).toUpperCase()),
    ),
  ).sort();
  const failures: string[] = [];
  if (contract.requiredTools.includes("web_search") && searches.length === 0) failures.push("missing_search");
  if (contract.totalUrls && fetches.length < contract.totalUrls) failures.push("missing_fetch_quota");
  const drifted = [...searches, ...fetches]
    .filter((event) => event.entity && lockedEntities.length > 0)
    .map((event) => String(event.entity).toUpperCase())
    .filter((entity) => !lockedEntities.includes(entity));
  if (drifted.length > 0) failures.push(`entity_drift:${Array.from(new Set(drifted)).sort().join(",")}`);
  const target = Math.max(1, contract.requiredTools.length + (contract.totalUrls ? 1 : 0));
  return {
    ok: failures.length === 0,
    score: Math.max(0, Number(((target - failures.length) / target).toFixed(3))),
    failures,
    metrics: {
      searchCount: searches.length,
      fetchCount: fetches.length,
      requiredFetchCount: contract.totalUrls,
      lockedEntities,
    },
  };
}

export function makePlainTrace(): TraceEvent[] {
  return [
    { kind: "entity_locked", entity: "ROKU", summary: "Locked top mover" },
    { kind: "entity_locked", entity: "TBN", summary: "Locked top mover" },
    { kind: "entity_locked", entity: "SENEA", summary: "Locked top mover" },
    { tool: "web_search", entity: "EPAM", ok: true, summary: "Planner drifted to unrelated ticker" },
    { tool: "web_fetch", entity: "ROKU", url: "https://source.test/ROKU/0", ok: true, summary: "Fetched one source" },
  ];
}

export function makeHarnessTrace(contract: SourceContract): TraceEvent[] {
  const entities = contract.entityCount === 3 ? ["ROKU", "TBN", "SENEA"] : ["A", "B", "C"].slice(0, contract.entityCount || 3);
  const events: TraceEvent[] = entities.map((entity) => ({
    kind: "entity_locked",
    entity,
    summary: "Locked entity before source collection",
  }));
  entities.forEach((entity) => {
    events.push({ tool: "web_search", entity, ok: true, summary: `Searched ${entity}` });
  });
  entities.forEach((entity) => {
    for (let index = 0; index < (contract.urlsPerEntity || 1); index += 1) {
      events.push({
        tool: "web_fetch",
        entity,
        url: `https://source.test/${entity}/${index}`,
        ok: true,
        summary: `Fetched source ${index + 1} for ${entity}`,
      });
    }
  });
  return events;
}

export function decide(contract: SourceContract, evalReport: TraceEval): KernelDecision {
  if (contract.totalUrls && evalReport.metrics.fetchCount >= contract.totalUrls) {
    return { state: "respond", reason: "source quota met" };
  }
  if (evalReport.metrics.searchCount === 0) {
    return { state: "act", reason: "need entity discovery/search", nextTool: "web_search", nextArgs: { query: discoveryQuery(contract.objective) } };
  }
  return { state: "act", reason: "need selected URL fetches", nextTool: "web_fetch", nextArgs: { url: "<selected_url>" } };
}
