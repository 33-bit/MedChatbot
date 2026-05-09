"""Internal exceptions used to stop unsafe degraded answers."""


class DependencyUnavailable(RuntimeError):
    """A required external dependency is unavailable for this turn."""


class Neo4jUnavailable(DependencyUnavailable):
    """Neo4j knowledge graph access failed."""


class QdrantUnavailable(DependencyUnavailable):
    """Qdrant vector retrieval access failed."""
