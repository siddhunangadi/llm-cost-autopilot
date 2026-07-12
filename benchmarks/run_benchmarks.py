"""Reproducible benchmark suite validating the performance/failover claims
in CLAUDE.md's Performance Targets and Portfolio Completion sections.

Run from the repo root: python -m benchmarks.run_benchmarks
Writes benchmarks/report.md.

Reuses the real production RoutingEngine/ModelRegistry/ProviderExecutor
wiring (same classes and config files main.py uses) with MockProvider
swapped in for every provider, so timings measure this codebase's actual
routing/classification overhead with zero network calls -- not a
reimplementation of the routing stack.
"""

import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import HeuristicComplexityClassifier
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.providers.base import BaseProvider, ProviderError
from backend.providers.circuit_breaker import CircuitBreaker, CircuitState
from backend.providers.executor import CircuitOpenError, ProviderExecutor
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.routing.config_loader import RoutingConfigLoader
from backend.routing.engine import RoutingEngine
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import (
    BalancedStrategy, CostOptimizedStrategy, LatencyOptimizedStrategy, QualityOptimizedStrategy,
)
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.credential_store import CredentialStore
from backend.services.model_registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTING_LATENCY_TARGET_MS = 50.0
CLASSIFIER_LATENCY_TARGET_MS = 10.0

PROMPTS = [
    "List three fruits.",
    "What's the capital of France?",
    "Summarize this in one sentence: the sky is blue.",
    "Write a haiku about autumn.",
    (
        "Analyze and compare these two algorithms, explain the reasoning step by "
        "step, calculate their time complexity, and format the answer as bullet "
        "points. You must include examples and should ensure correctness."
    ),
    (
        "Design a distributed rate limiter that works across multiple regions, "
        "explain the tradeoffs between token bucket and sliding window "
        "approaches, and provide pseudocode with error handling for the edge "
        "cases where the coordination service is temporarily unreachable."
    ),
    "Translate 'good morning' to Spanish.",
    "Fix this code: def add(a, b) return a + b",
    "Explain quantum entanglement to a five-year-old.",
    "What's 15% of 240?",
]


def _build_routing_stack(tmp_path: Path) -> tuple[RoutingEngine, ModelRegistry]:
    """Same wiring as api/main.py's lifespan(), against the real
    backend/config/models.yaml and routing.yaml, with every provider
    faked via a Settings API key and MockProvider swapped in for the
    factory -- no network calls, no real credentials needed."""
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/bench.db",
        openai_api_key="sk-bench", anthropic_api_key="sk-bench",
        ollama_base_url="http://localhost:11434", gemini_api_key="sk-bench",
        nvidia_nim_api_key="sk-bench", openrouter_api_key="sk-bench",
        groq_api_key="sk-bench", mistral_api_key="sk-bench",
    )
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    factory = ProviderFactory()
    factory.register("mock", MockProvider, user_configurable=False)
    for name in ("openai", "anthropic", "ollama", "gemini", "nvidia_nim", "openrouter", "groq", "mistral"):
        factory.register(name, MockProvider)
    credential_store = CredentialStore(
        session_factory=session_factory, settings=settings, provider_names=factory.registered_names(),
    )
    provider_manager = ProviderManager(factory, credential_store)

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(REPO_ROOT / "backend/config/models.yaml"),
    )
    model_registry.reload()

    routing_config = RoutingConfigLoader.load(str(REPO_ROOT / "backend/config/routing.yaml"))
    routing_engine = RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=HeuristicComplexityClassifier(routing_config.classifier),
        routing_policy=RoutingPolicy(routing_config.policy),
        strategies={
            "cost": CostOptimizedStrategy(),
            "latency": LatencyOptimizedStrategy(),
            "quality": QualityOptimizedStrategy(),
            "balanced": BalancedStrategy(routing_config.balanced_strategy),
        },
        explanation_generator=ExplanationGenerator(),
    )
    return routing_engine, model_registry


def _percentile(values: list[float], p: float) -> float:
    return statistics.quantiles(values, n=100)[int(p) - 1] if len(values) > 1 else values[0]


def benchmark_routing_latency(engine: RoutingEngine, iterations: int = 300) -> dict:
    samples_ms = []
    for i in range(iterations):
        prompt = PROMPTS[i % len(PROMPTS)]
        start = time.perf_counter()
        engine.route(prompt, strategy_name="balanced")
        samples_ms.append((time.perf_counter() - start) * 1000)

    avg = statistics.mean(samples_ms)
    p95 = _percentile(samples_ms, 95)
    return {
        "iterations": iterations, "avg_ms": avg, "p95_ms": p95,
        "target_ms": ROUTING_LATENCY_TARGET_MS, "passed": p95 < ROUTING_LATENCY_TARGET_MS,
    }


def benchmark_classifier_latency(iterations: int = 300) -> dict:
    from backend.routing.config import ClassifierPolicy

    analyzer = PromptAnalyzer()
    classifier = HeuristicComplexityClassifier(ClassifierPolicy(simple_max=1, medium_max=3))
    samples_ms = []
    for i in range(iterations):
        prompt = PROMPTS[i % len(PROMPTS)]
        features = analyzer.analyze(prompt)
        start = time.perf_counter()
        classifier.classify(features)
        samples_ms.append((time.perf_counter() - start) * 1000)

    avg = statistics.mean(samples_ms)
    p95 = _percentile(samples_ms, 95)
    return {
        "iterations": iterations, "avg_ms": avg, "p95_ms": p95,
        "target_ms": CLASSIFIER_LATENCY_TARGET_MS, "passed": p95 < CLASSIFIER_LATENCY_TARGET_MS,
    }


def run_load_test(engine: RoutingEngine, model_registry: ModelRegistry, request_count: int = 500) -> dict:
    models = model_registry.get_available_models()
    baseline = max(models, key=lambda m: m.input_cost + m.output_cost)
    analyzer = PromptAnalyzer()

    actual_cost = 0.0
    baseline_cost = 0.0
    distribution: dict[str, int] = {}
    quality_scores = []
    baseline_quality_scores = []

    for i in range(request_count):
        prompt = PROMPTS[i % len(PROMPTS)]
        decision = engine.route(prompt, strategy_name="balanced")
        actual_cost += decision.estimated_cost
        distribution[decision.selected_model] = distribution.get(decision.selected_model, 0) + 1
        selected_spec = model_registry.get_model(decision.selected_model)
        quality_scores.append(selected_spec.benchmark_score)
        baseline_quality_scores.append(baseline.benchmark_score)

        features = analyzer.analyze(prompt)
        baseline_cost += model_registry.estimate_cost(
            baseline.id, features.estimated_tokens, features.estimated_output_tokens
        )

    savings = baseline_cost - actual_cost
    savings_percent = savings / baseline_cost if baseline_cost else 0.0
    quality_parity = (
        statistics.mean(quality_scores) / statistics.mean(baseline_quality_scores)
        if baseline_quality_scores else 1.0
    )
    return {
        "request_count": request_count,
        "baseline_model": baseline.id,
        "actual_cost": actual_cost,
        "baseline_cost": baseline_cost,
        "savings": savings,
        "savings_percent": savings_percent,
        "quality_parity_percent": quality_parity * 100,
        "distribution": dict(sorted(distribution.items(), key=lambda kv: -kv[1])),
    }


class _FailThenSucceedProvider(BaseProvider):
    """Fails a fixed number of times, then succeeds -- used to drive a
    real CircuitBreaker through open -> half-open -> closed and capture
    the transitions, rather than asserting failover exists without
    evidence."""

    def __init__(self, fail_count: int) -> None:
        self._remaining_failures = fail_count

    @property
    def name(self) -> str:
        return "flaky"

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ProviderError("simulated provider outage")
        return "recovered"

    async def stream(self, prompt: str, model: str, **kwargs):
        yield await self.generate(prompt, model, **kwargs)

    async def health_check(self) -> bool:
        return self._remaining_failures == 0

    def count_tokens(self, text: str) -> int:
        return len(text)

    def estimate_cost(self, input_tokens, output_tokens, input_cost, output_cost) -> float:
        return 0.0


class _StubProviderManager:
    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    def get_provider(self, name: str) -> BaseProvider:
        return self._provider


async def run_failover_demo(open_timeout_seconds: float = 0.3) -> dict:
    failure_threshold = 3
    provider = _FailThenSucceedProvider(fail_count=failure_threshold)
    breaker = CircuitBreaker(failure_threshold=failure_threshold, open_timeout=open_timeout_seconds)
    executor = ProviderExecutor(
        provider_manager=_StubProviderManager(provider),
        retry_policy=None,  # unused: retry=False below skips the retry policy entirely
        circuit_breakers={"flaky": breaker},
        event_bus=EventBus(),
    )

    transitions = [f"start: {breaker.state.value}"]
    for _ in range(failure_threshold):
        try:
            await executor.generate("flaky", "prompt", model="m", retry=False)
        except ProviderError:
            pass
    transitions.append(f"after {failure_threshold} failures: {breaker.state.value}")
    assert breaker.state == CircuitState.OPEN, "circuit did not open after failure_threshold failures"

    rejected = False
    try:
        await executor.generate("flaky", "prompt", model="m", retry=False)
    except CircuitOpenError:
        rejected = True
    transitions.append(f"request while open: {'rejected' if rejected else 'NOT REJECTED (bug)'}")

    await asyncio.sleep(open_timeout_seconds + 0.05)
    result = await executor.generate("flaky", "prompt", model="m", retry=False)
    transitions.append(f"after recovery probe: {breaker.state.value} (result={result!r})")
    assert breaker.state == CircuitState.CLOSED, "circuit did not recover after successful probe"

    return {
        "failure_threshold": failure_threshold,
        "open_timeout_seconds": open_timeout_seconds,
        "rejected_while_open": rejected,
        "recovered": breaker.state == CircuitState.CLOSED,
        "transitions": transitions,
    }


def _fmt_pass(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _render_report(routing: dict, classifier: dict, load_test: dict, failover: dict) -> str:
    lines = [
        "# Benchmark Report",
        "",
        "Generated by `benchmarks/run_benchmarks.py`. All numbers below come",
        "from actually running this codebase's `RoutingEngine`,",
        "`HeuristicComplexityClassifier`, and `ProviderExecutor`/`CircuitBreaker`",
        "against production config with `MockProvider` standing in for network",
        "calls -- not hand-written estimates.",
        "",
        "## Routing Latency",
        "",
        f"- Iterations: {routing['iterations']}",
        f"- Average: {routing['avg_ms']:.2f} ms",
        f"- P95: {routing['p95_ms']:.2f} ms",
        f"- Target: < {routing['target_ms']:.0f} ms",
        f"- **{_fmt_pass(routing['passed'])}**",
        "",
        "## Classifier Latency",
        "",
        f"- Iterations: {classifier['iterations']}",
        f"- Average: {classifier['avg_ms']:.3f} ms",
        f"- P95: {classifier['p95_ms']:.3f} ms",
        f"- Target: < {classifier['target_ms']:.0f} ms",
        f"- **{_fmt_pass(classifier['passed'])}**",
        "",
        "## Load Test",
        "",
        f"- Requests: {load_test['request_count']}",
        f"- Baseline model (highest combined cost): {load_test['baseline_model']}",
        f"- Actual cost: ${load_test['actual_cost']:.4f}",
        f"- Baseline cost: ${load_test['baseline_cost']:.4f}",
        f"- Savings: ${load_test['savings']:.4f} ({load_test['savings_percent']:.1%})",
        f"- Quality parity: {load_test['quality_parity_percent']:.1f}%",
        "  (ratio of configured model `benchmark_score` ratings, selected vs. baseline --",
        "   not a live LLM-judge or VerificationService pass rate; see dashboard Quality",
        "   metrics for measured-output-quality data.)",
        "- Routing distribution:",
    ]
    for model, count in load_test["distribution"].items():
        lines.append(f"  - {model}: {count} ({count / load_test['request_count']:.1%})")
    lines += [
        "",
        "## Provider Failover",
        "",
        f"- Failure threshold: {failover['failure_threshold']}",
        f"- Open timeout: {failover['open_timeout_seconds']}s",
        f"- Requests rejected while circuit open: {failover['rejected_while_open']}",
        f"- Recovered after timeout + successful probe: {failover['recovered']}",
        "- Transition log:",
    ]
    for t in failover["transitions"]:
        lines.append(f"  - {t}")
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        engine, model_registry = _build_routing_stack(tmp_path)

        routing = benchmark_routing_latency(engine)
        classifier = benchmark_classifier_latency()
        load_test = run_load_test(engine, model_registry)
        failover = await run_failover_demo()

    report = _render_report(routing, classifier, load_test, failover)
    report_path = REPO_ROOT / "benchmarks" / "report.md"
    report_path.write_text(report)
    print(report)
    print(f"\nWritten to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
