from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Router imports
from .routers import reactions, explanation, coord3d, voice, tutor

# ML model initializer
from .core.model_loader import load_model


def create_app() -> FastAPI:
    app = FastAPI(
        title="ReactAIvate ML API",
        description="GNN + LLM reaction mechanism predictor backend",
        version="1.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Startup: load ML model
    @app.on_event("startup")
    def startup_event():
        print(" Loading ML model...")
        load_model()
        print("ML Model Loaded Successfully")

    # Register Routers
    app.include_router(reactions.router, prefix="/api/v1")
    app.include_router(explanation.router, prefix="/api/v1")
    app.include_router(coord3d.router, prefix="/api/v1")
    app.include_router(voice.router, prefix="/api/v1")
    app.include_router(tutor.router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
