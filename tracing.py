"""LangSmith / OpenTelemetry tracing bootstrap."""

import os

from logging_config import logger


def initialize_tracing() -> None:
    """Initialize LangSmith + OpenInference tracing from environment variables."""
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if not tracing_enabled:
        logger.info("Tracing disabled (LANGSMITH_TRACING is not true)")
        return

    project_name = os.getenv("LANGSMITH_PROJECT", "autogen-login-agent").strip().strip('"')

    try:
        # Compatibility patch: newer OpenTelemetry SDK calls _on_ending on span processors.
        # Some LangSmith versions expose OtelSpanProcessor without this method.
        from langsmith.integrations.otel.processor import OtelSpanProcessor
        if not hasattr(OtelSpanProcessor, "_on_ending"):
            def _on_ending(self, span):
                return None

            OtelSpanProcessor._on_ending = _on_ending

        from langsmith.integrations.otel import configure
        configure(project_name=project_name)

        tracing_parts_enabled: list[str] = ["langsmith-otel"]

        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
            tracing_parts_enabled.append("openai")
        except Exception as e:
            logger.warning(f"OpenAI instrumentation not enabled: {e}")

        autogen_tracing = os.getenv("LANGSMITH_AUTOGEN_TRACING", "false").lower() == "true"
        if autogen_tracing:
            try:
                from openinference.instrumentation.autogen import AutogenInstrumentor
                AutogenInstrumentor().instrument()
                tracing_parts_enabled.append("autogen")
            except Exception as e:
                logger.warning(f"AutoGen instrumentation not enabled: {e}")

        logger.info(
            f"Tracing enabled ({', '.join(tracing_parts_enabled)}), project={project_name}"
        )
    except ImportError as e:
        logger.info(f"Tracing disabled (optional tracing dependencies are missing): {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize tracing: {e}")
