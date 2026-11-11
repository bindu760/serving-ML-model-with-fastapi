"""Health check route."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    """Report service health and model availability."""
    return {"status": "ok", "model_loaded": getattr(request.app.state, "model_service", None) is not None}
