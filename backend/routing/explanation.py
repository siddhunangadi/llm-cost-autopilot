from backend.classifier.complexity_classifier import ClassificationResult
from backend.routing.context import RoutingContext
from backend.services.model_registry import ModelSpec


class ExplanationGenerator:
    def generate(
        self,
        context: RoutingContext,
        selected: ModelSpec,
        strategy_name: str,
        classification: ClassificationResult,
    ) -> list[str]:
        if classification.signals:
            signal_text = ", ".join(classification.signals)
            classification_line = (
                f"Classified as {classification.tier.value} "
                f"(confidence {classification.confidence}): {signal_text}."
            )
        else:
            classification_line = (
                f"Classified as {classification.tier.value} "
                f"(confidence {classification.confidence}): no complexity signals detected."
            )

        return [
            classification_line,
            f"Strategy '{strategy_name}' evaluated {len(context.candidates)} eligible model(s).",
            f"Selected '{selected.id}'.",
        ]
