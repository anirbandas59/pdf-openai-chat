from app.chat.tracing.langfuse import langfuse
from langfuse.model import CreateTrace


class TraceableChain:
    def __call__(self, *args, **kwargs):
        trace = langfuse.trace(
            CreateTrace(
                id=self.metadata["conversation_id"],
                metadata=self.metadata
            )
        )

        callbacks = kwargs.get("callbacks", [])
        callbacks.append(trace.getNewHandler())
        kwargs["callbacks"] = callbacks

        return super().__call__(*args, **kwargs)
