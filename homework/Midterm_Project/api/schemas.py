from typing import Literal, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    pais_o_d: str = Field(..., min_length=3, max_length=3, examples=["USA"])
    tipo: Literal["Exportaciones", "Importaciones"] = Field(..., examples=["Exportaciones"])
    tipo_bien: Literal["Consumo", "Intermedio", "Capital"] = Field(..., examples=["Intermedio"])
    anio: int = Field(..., ge=2015, le=2030, examples=[2026])
    mes: int = Field(..., ge=1, le=12, examples=[3])


class PredictResponse(BaseModel):
    predicted_val_usd_miles: float
    pais_o_d: str
    clave_continente: Optional[str] = None
    clave_region: Optional[str] = None
    nombre: Optional[str] = None              # ← añadir
    nombre_continente: Optional[str] = None   # ← añadir
    nombre_region: Optional[str] = None       # ← añadir
    tipo: str
    tipo_bien: str
    anio: int
    mes: int
    model: str
    version: str
    warning: Optional[str] = None


class PaisInfo(BaseModel):
    pais_o_d: str
    clave_continente: Optional[str] = None
    clave_region: Optional[str] = None
    nombre: Optional[str] = None
    nombre_continente: Optional[str] = None
    nombre_region: Optional[str] = None


class MetadataResponse(BaseModel):
    model: str
    version: str
    n_paises: int
    metrics_val: dict