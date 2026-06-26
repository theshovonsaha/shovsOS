import {
  decide,
  evaluateTrace,
  inferSourceContract,
  makeHarnessTrace,
  makePlainTrace,
  type SourceContract,
  type TraceEvent,
  type TraceEval,
} from "./harness";
import "./styles.css";

const examples = [
  "Search top 3 stocks today, then search each, fetch 3 URLs each.",
  "Find top 3 sushi places in Toronto, search each, fetch 3 URLs each.",
  "Find top three laptops for students, search each, fetch three articles each.",
];

let objective = examples[0];
let selectedRun: "plain" | "harness" = "harness";

function render() {
  const contract = inferSourceContract(objective);
  const plainTrace = makePlainTrace();
  const harnessTrace = makeHarnessTrace(contract);
  const plainEval = evaluateTrace(contract, plainTrace);
  const harnessEval = evaluateTrace(contract, harnessTrace);
  const activeTrace = selectedRun === "plain" ? plainTrace : harnessTrace;
  const activeEval = selectedRun === "plain" ? plainEval : harnessEval;
  const decision = decide(contract, activeEval);
  const root = document.querySelector<HTMLDivElement>("#root");
  if (!root) return;
  root.innerHTML = `
    <main class="shell">
      <section class="hero">
        <div>
          <p class="eyebrow">Shovs Harness Core</p>
          <h1>Test the agent run, not the final prose.</h1>
          <p class="subhead">
            Paste a task and see how a small harness compiles requirements, catches drift,
            and decides whether the run can answer.
          </p>
        </div>
        <div class="heroStats" aria-label="Current harness score">
          <span>${Math.round(harnessEval.score * 100)}%</span>
          <small>harness trace score</small>
        </div>
      </section>

      <section class="workspace">
        <aside class="controlPanel">
          <label class="fieldLabel" for="objective">Task</label>
          <textarea id="objective">${escapeHtml(objective)}</textarea>
          <div class="examples">
            ${examples
              .map((example, index) => `<button type="button" data-example="${index}">${escapeHtml(example.split(",")[0])}</button>`)
              .join("")}
          </div>
          ${card("Inferred Contract", metrics(contract))}
        </aside>

        <section class="runPanel">
          <div class="tabs" role="tablist" aria-label="Run comparison">
            <button class="${selectedRun === "plain" ? "active" : ""}" type="button" data-run="plain">Plain loop</button>
            <button class="${selectedRun === "harness" ? "active" : ""}" type="button" data-run="harness">Harness loop</button>
          </div>
          ${scorecard(selectedRun === "plain" ? "Plain loop" : "Harness loop", activeEval)}
          ${card(
            "Kernel Decision",
            `<div class="decision ${decision.state}">
              <strong>${decision.state}</strong>
              <span>${escapeHtml(decision.reason)}</span>
            </div>
            ${decision.nextTool ? `<code>${escapeHtml(`${decision.nextTool} ${JSON.stringify(decision.nextArgs)}`)}</code>` : ""}`,
          )}
          ${timeline(activeTrace)}
        </section>

        <aside class="detailPanel">
          ${card(
            "Why This Matters",
            `<p>The model can suggest work. The ledger decides what happened.
            The trace eval decides whether the answer is allowed.</p>`,
          )}
          ${card("Raw State", `<pre>${escapeHtml(JSON.stringify({ contract, activeEval, decision }, null, 2))}</pre>`)}
        </aside>
      </section>
    </main>
  `;
  bindEvents();
}

function bindEvents() {
  document.querySelector<HTMLTextAreaElement>("#objective")?.addEventListener("input", (event) => {
    objective = (event.target as HTMLTextAreaElement).value;
    render();
  });
  document.querySelectorAll<HTMLButtonElement>("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      objective = examples[Number(button.dataset.example ?? 0)] ?? examples[0];
      render();
    });
  });
  document.querySelectorAll<HTMLButtonElement>("[data-run]").forEach((button) => {
    button.addEventListener("click", () => {
      selectedRun = button.dataset.run === "plain" ? "plain" : "harness";
      render();
    });
  });
}

function card(title: string, body: string) {
  return `<section class="card"><h2>${escapeHtml(title)}</h2>${body}</section>`;
}

function metrics(contract: SourceContract) {
  return [
    metric("Entities", contract.entityCount || "not recorded"),
    metric("URLs each", contract.urlsPerEntity || "not recorded"),
    metric("Total fetches", contract.totalUrls || "not recorded"),
    metric("Tools", contract.requiredTools.join(", ") || "none"),
  ].join("");
}

function metric(label: string, value: string | number) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function scorecard(label: string, report: TraceEval) {
  return `
    <section class="scorecard ${report.ok ? "pass" : "fail"}">
      <div>
        <p>${escapeHtml(label)}</p>
        <strong>${Math.round(report.score * 100)}%</strong>
      </div>
      <div>
        <span>${report.metrics.fetchCount}/${report.metrics.requiredFetchCount || 0} fetches</span>
        <span>${escapeHtml(report.failures.length ? report.failures.join(" · ") : "state checks passed")}</span>
      </div>
    </section>
  `;
}

function timeline(events: TraceEvent[]) {
  return `
    <section class="timeline" aria-label="Trace timeline">
      <h2>Trace Timeline</h2>
      ${events
        .map(
          (event) => `
            <article class="traceRow">
              <span class="dot"></span>
              <div>
                <header>
                  <strong>${escapeHtml(event.kind ?? event.tool ?? "event")}</strong>
                  <span>${escapeHtml(event.entity ?? "not recorded")}</span>
                </header>
                <p>${escapeHtml(event.summary ?? event.url ?? "No summary recorded")}</p>
                ${event.url ? `<code>${escapeHtml(event.url)}</code>` : ""}
              </div>
            </article>
          `,
        )
        .join("")}
    </section>
  `;
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

render();
