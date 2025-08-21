from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.api.websocket import ws_router
from app.api.sse import sse_router
from app.api.enhanced_routes import enhanced_router
from app.api.evaluation_routes import evaluation_router
from app.admin.routes import admin_router
from app.observability.logging import configure_logging
from app.observability.metrics import register_metrics
from app.security.middleware import SecurityMiddleware
from app.models.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Advanced Multi-Modal RAG",
        version="0.1.0",
        description="Production-grade multi-modal RAG with dynamic model adaptation",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityMiddleware)

    configure_logging(settings.log_level)
    register_metrics(app)

    app.include_router(api_router, prefix="/api")
    app.include_router(enhanced_router, prefix="/api")
    app.include_router(evaluation_router, prefix="/api/evaluation")
    app.include_router(admin_router, prefix="/api")
    app.include_router(ws_router)
    app.include_router(sse_router, prefix="/api")

    return app


app = create_app()


@app.get("/")
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/api/admin/dashboard", status_code=307)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


