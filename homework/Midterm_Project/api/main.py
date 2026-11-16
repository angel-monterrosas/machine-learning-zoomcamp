"""API FastAPI para el predictor BCMM."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    MetadataResponse, PaisInfo, PredictRequest, PredictResponse,
)
from src.predict import get_predictor

# Configuración de logs
logger = logging.getLogger("uvicorn.error")

# Variable global para mantener el modelo en memoria
predictor_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo en memoria al iniciar y lo libera al apagar."""
    global predictor_service
    logger.info("Iniciando BCMM Predictor API...")
    logger.info("Cargando modelo y artefactos en memoria...")
    try:
        predictor_service = get_predictor()
        logger.info(
            f"Modelo '{predictor_service.winner_name}' v{predictor_service.version} "
            f"cargado correctamente ({len(predictor_service.pais_lookup)} países)."
        )
    except Exception as e:
        logger.critical(f"Error fatal al cargar el modelo: {e}", exc_info=True)
        raise e

    yield  # Aquí es donde la API recibe peticiones

    logger.info("Apagando BCMM Predictor API...")
    predictor_service = None


app = FastAPI(
    title="BCMM Predictor API",
    description="API REST para la predicción del comercio exterior de México (INEGI BCMM).",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: ["https://tu-dominio.com"]
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Helper: validar que el modelo esté cargado ---
def _require_predictor():
    """Lanza 503 si el modelo no está cargado en memoria."""
    if predictor_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está cargado. Intente de nuevo en unos segundos.",
        )
    return predictor_service


# --- Manejador global de errores ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura cualquier error no controlado y devuelve un JSON limpio."""
    logger.error(f"Error no controlado en {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor. Por favor, contacte al administrador."},
    )


# --- Endpoints ---

@app.get("/health", tags=["Sistema"])
def health():
    """Verifica que el servicio esté en línea y el modelo cargado."""
    if predictor_service is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Modelo no cargado"},
        )
    return {
        "status": "healthy",
        "model": predictor_service.winner_name,
        "version": predictor_service.version,
    }


@app.get("/metadata", response_model=MetadataResponse, tags=["Sistema"])
def metadata():
    """Devuelve los metadatos del modelo en producción."""
    p = _require_predictor()
    return p.metadata()


@app.get("/paises", response_model=list[PaisInfo], tags=["Catálogos"])
def list_paises():
    """Devuelve el catálogo de países disponibles para predecir."""
    p = _require_predictor()
    return p.list_paises()


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
def predict(request: PredictRequest):
    """
    Genera una predicción de comercio exterior.

    - **pais_o_d**: Código ISO de 3 letras del país.
    - **tipo**: Exportaciones o Importaciones.
    - **tipo_bien**: Consumo, Intermedio o Capital.
    - **anio**: Año de la predicción.
    - **mes**: Mes de la predicción (1-12).
    """
    p = _require_predictor()
    start = time.perf_counter()

    try:
        result = p.predict_one(**request.model_dump())
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Predicción exitosa: {request.pais_o_d} {request.tipo} {request.tipo_bien} "
            f"({request.anio}-{request.mes:02d}) en {elapsed_ms:.1f} ms"
        )
        return result
    except ValueError as e:
        logger.warning(f"Error de validación en /predict: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Fallo en el modelo predictivo: {str(e)}", exc_info=True)
        raise e