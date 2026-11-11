"""Model metadata route."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info")
def model_info(request: Request) -> dict[str, object]:
    """Return metadata for the active model."""
    metadata = request.app.state.model_service.metadata
    return {name: metadata.get(name) for name in ("version", "metrics", "trained_at", "products")}
