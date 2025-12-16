import json
import os
from typing import List, Optional, Dict, Any, TypeVar, Generic, Literal, Union
from fastapi import FastAPI, HTTPException, Query, Body, status
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from datetime import datetime, timedelta
from fastapi.responses import HTMLResponse
from unidecode import unidecode

# --- CONFIGURACIÓN BASE DE DATOS ---
MONGO_URI = "mongodb://root:1AA3Ct23s@panelofertas.omexan.com:27017/?tls=false"
client = MongoClient(MONGO_URI)
db = client["ofertasERP"]
users_collection = db["users"]
orders_collection = db[
    "orders"
]  # Renombrado a orders_collection para evitar conflicto con el modelo


# --- Funciones Auxiliares ---
def convert_date_format(date_str: Optional[str]) -> Optional[str]:
    """
    Retorna la fecha en formato YYYY-MM-DD si es válida, o None.
    Asume que el input ya viene en ese formato o similar compatible (ISO 8601).
    """
    if not date_str:
        return None
    try:
        # Intenta parsear y luego formatear para asegurar consistencia
        # .replace('Z', '+00:00') para manejar fechas ISO con Z al final (UTC)
        date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        return None  # Si no es un formato válido, retorna None


# --- MODELOS Pydantic ---

# Modelo genérico para respuestas paginadas
T = TypeVar("T")


# --- Modelo para la petición ---
class MongoOperation(BaseModel):
    collection_name: str = Field(
        "orders", description="Nombre de la colección de MongoDB."
    )

    # Tipo de operación: find, insert_one, update_one, delete_one, aggregate
    operation: Literal[
        "find", "insert_one", "update_one", "delete_one", "aggregate"
    ] = Field(..., description="Tipo de operación a realizar en MongoDB.")

    # Campos comunes para varias operaciones
    # 'filter' se usa en find, update_one, delete_one
    filter: Dict[str, Any] = Field(
        default_factory=dict,
        description="Documento de filtro para la operación (e.g., {'_id': 'some_id'}).",
    )

    # 'document' se usa en insert_one
    document: Dict[str, Any] = Field(
        default_factory=dict, description="Documento a insertar (para insert_one)."
    )

    # 'update' se usa en update_one (debe contener operadores como $set, $inc, etc.)
    update: Dict[str, Any] = Field(
        default_factory=dict,
        description="Documento con operadores de actualización (e.g., {'$set': {'field': 'value'}}).",
    )

    # 'projection' se usa en find
    projection: Union[Dict[str, Any], None] = Field(
        None, description="Documento de proyección para 'find' (e.g., {'field': 1})."
    )

    # Otros parámetros para 'find'
    limit: int = Field(
        0,
        description="Límite de documentos a retornar para 'find'. 0 significa sin límite.",
    )
    sort: Union[List[tuple], None] = Field(
        None,
        description="Criterio de ordenación para 'find' (e.g., [('field', 1), ('other_field', -1)]).",
    )
    skip: int = Field(0, description="Número de documentos a omitir para 'find'.")

    # 'pipeline' se usa en aggregate
    pipeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Array de etapas para la pipeline de agregación.",
    )


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total_count: int
    page: int
    page_size: int


class CreateLeadRequest(BaseModel):
    order_id: int = Field(..., description="ID de la orden de Shopify")
    kommo_order_number: str = Field(..., description="Número de orden de Kommo")
    nota: str = Field(..., description="Motivo o descripción para el lead")
    email: str = Field(..., description="Correo del usuario que crea el lead")


class OrderFlagsUpdate(BaseModel):
    is_deudor: Optional[bool] = None
    is_blacklist: Optional[bool] = None


class SearchFilters(BaseModel):
    # Grupo: Cliente
    order_number: Optional[str] = Field(None, description="Buscar por numero de orden")
    partialName: Optional[str] = Field(
        None, description="Buscar por parte del nombre o email"
    )
    phone: Optional[str] = Field(None, description="Buscar teléfono")
    email: Optional[str] = Field(None, description="Buscar email")
    shopifyId: Optional[int] = Field(None, description="ID de la orden en Shopify")
    customerId: Optional[int] = Field(None, description="ID del cliente en Shopify")
    # Grupo: Dirección
    street: Optional[str] = Field(None, description="Buscar en la dirección de envío")
    city: Optional[str] = Field(None, description="Buscar por ciudad")
    postalCode: Optional[str] = Field(None, description="Código postal")
    # Grupo: Personal y Origen
    collectorName: Optional[str] = Field(None, description="Nombre del cobrador")
    confirmerName: Optional[str] = Field(None, description="Nombre del confirmador")
    origin: Optional[str] = Field(
        None, description="Origen de la compra (e.g., 'facebook')"
    )
    # Grupo: Estados
    status: Optional[int] = Field(None, description="Estado de envío (0-4)")
    paidStatus: Optional[str] = Field(None, description="Estado de pago")
    # Grupo: Fechas
    date: Optional[str] = Field(None, description="Fecha exacta (YYYY-MM-DD)")
    dateFrom: Optional[str] = Field(None, description="Fecha de inicio para rango")
    dateTo: Optional[str] = Field(None, description="Fecha de fin para rango")

    dateConfirmacion: Optional[str] = Field(
        None, description="Fecha exacta (YYYY-MM-DD)"
    )
    dateFromConfirmacion: Optional[str] = Field(
        None, description="Fecha de inicio para rango"
    )
    dateToConfirmacion: Optional[str] = Field(
        None, description="Fecha de fin para rango"
    )

    dateEntrega: Optional[str] = Field(None, description="Fecha exacta (YYYY-MM-DD)")
    dateFromEntrega: Optional[str] = Field(
        None, description="Fecha de inicio para rango"
    )
    dateToEntrega: Optional[str] = Field(None, description="Fecha de fin para rango")

    datePago: Optional[str] = Field(None, description="Fecha exacta (YYYY-MM-DD)")
    dateFromPago: Optional[str] = Field(None, description="Fecha de inicio para rango")
    dateToPago: Optional[str] = Field(None, description="Fecha de fin para rango")

    # Grupo: Producto
    product: Optional[str] = Field(None, description="Buscar por nombre del producto")


class ClientOrderSummary(BaseModel):
    id: int
    kommo_order_number: str
    fecha_compra: str
    total: float
    status: int
    paid_status: str


class Client(BaseModel):
    id: int  # Shopify Customer ID
    name: str
    email: str
    phone: Optional[str] = None
    totalSpent: float
    orderCount: int
    userType: str  # e.g., "client", "repeat"
    lastOrderDate: Optional[str] = None
    orders: List[ClientOrderSummary] = []  # Se mantiene para el modal de cliente


class DashboardMetrics(BaseModel):
    totalOrders: int
    revenue: float
    unconfirmedOrders: int
    confirmedOrders: int
    inTransitAndSoon: int
    deliveredAndPaid: int
    deliveredAndUnpaid: int
    canceledOrders: int
    averageOrderValue: float
    deliveryRate: float


class MinimalOrder(BaseModel):
    id: int  # shopify_order_id
    kommo_order_number: str
    fecha_compra: str
    status: int  # Numérico, frontend lo mapea
    paid_status: str
    total: float
    customerName: str
    origen_compra: str
    contactInfo: Dict[str, str]
    # products: List[Any] # No incluir productos en la vista mínima para tabla


class FullOrderDetails(BaseModel):
    id: int  # shopify_order_id
    kommo_order_number: str
    kommoLeadId: Optional[str] = None
    kommoContactId: Optional[str] = None
    fecha_compra: str
    fecha_confirmacion: Optional[str] = None
    fecha_entrega: Optional[str] = None
    fecha_pago: Optional[str] = None
    tipo_pago: Optional[str] = None
    tracking_code: Optional[str] = None
    paqueteria: Optional[str] = None
    origen_compra: Optional[str] = None
    origen_contacto: Optional[str] = None
    confirmador_name: Optional[str] = None
    cobrador_name: Optional[str] = None
    NOTA: Optional[str] = None
    nota_n8n: Optional[str] = None
    OBSERVACIONES: Optional[str] = None
    pipeline: Optional[str] = None
    raw_pipeline_stages: List[Dict[str, str]] = []
    status: int
    paidStatus: str
    user_type: Optional[str] = None
    order_type: Optional[str] = None
    profs_url: List[str] = []
    customerName: str
    total: float
    date: str
    products: List[Dict[str, Any]] = []
    basic_data: Dict[str, Any] = {}
    order_data: Dict[str, Any] = {}
    shipping_lines: List[Dict[str, Any]] = []
    billing_address: Dict[str, Any] = {}
    contactInfo: Dict[str, str] = {}


class ActivityLog(BaseModel):
    type: str
    message: str
    time: str
    icon: str
    color: str


class IntegrationStats(BaseModel):
    status: str
    connected_account: str
    stats: Dict[str, Any]


class Notification(BaseModel):
    id: int
    title: str
    message: str
    type: str
    time: str


class BlacklistUser(BaseModel):
    id: int
    name: str
    reason: str
    date: str


class Debtor(BaseModel):
    id: int
    name: str
    amount: float
    days_overdue: int


class ChartData(BaseModel):
    revenue_labels: List[str]
    revenue_current: List[float]
    revenue_previous: List[float]
    conversions_labels: List[str]
    conversions_data: List[int]
    top_products: List[Dict[str, Any]]


# Auth Models
class UserInDB(BaseModel):
    email: str
    password: str
    name: str
    role: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    email: str
    name: str
    role: str


# --- INICIALIZACIÓN APP ---
app = FastAPI(title="ERP Premium Backend Full Dynamic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Funciones Auxiliares para cálculo de crecimiento (se mantienen aunque no se usen en el nuevo DashboardMetrics) ---
def calculate_growth(current_value, previous_value):
    """Calcula el porcentaje de crecimiento. Maneja división por cero y crecimiento infinito."""
    if previous_value == 0:
        if current_value == 0:
            return 0.0
        else:
            return 1000.0
    return ((current_value - previous_value) / previous_value) * 100


# --- ENDPOINTS ---


@app.post("/api/v1/execute")
async def execute_mongo_query(request: MongoOperation):
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La conexión a la base de datos no está disponible.",
        )

    print(request)
    collection = db[request.collection_name]

    try:
        if request.operation == "find":
            cursor = collection.find(request.filter, request.projection)
            if request.sort:
                cursor = cursor.sort(request.sort)
            if request.skip > 0:
                cursor = cursor.skip(request.skip)
            if request.limit > 0:
                cursor = cursor.limit(request.limit)

            result = list(cursor)
            # Convertir ObjectId a string para una respuesta JSON válida
            for doc in result:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            return {
                "status": "success",
                "operation": "find",
                "data": result,
                "count": len(result),
            }

        elif request.operation == "insert_one":
            if not request.document:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El campo 'document' es requerido para insert_one.",
                )
            result = collection.insert_one(request.document)
            return {
                "status": "success",
                "operation": "insert_one",
                "inserted_id": str(result.inserted_id),
            }

        elif request.operation == "update_one":
            if not request.filter or not request.update:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Los campos 'filter' y 'update' son requeridos para update_one.",
                )
            result = collection.update_one(request.filter, request.update)
            return {
                "status": "success",
                "operation": "update_one",
                "matched_count": result.matched_count,
                "modified_count": result.modified_count,
                "upserted_id": str(result.upserted_id) if result.upserted_id else None,
            }

        elif request.operation == "delete_one":
            if not request.filter:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El campo 'filter' es requerido para delete_one.",
                )
            result = collection.delete_one(request.filter)
            return {
                "status": "success",
                "operation": "delete_one",
                "deleted_count": result.deleted_count,
            }

        elif request.operation == "aggregate":
            if not request.pipeline:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El campo 'pipeline' es requerido para aggregate.",
                )

            cursor = collection.aggregate(request.pipeline)
            result = list(cursor)
            # Convertir ObjectId a string para una respuesta JSON válida
            for doc in result:
                if "_id" in doc and not isinstance(
                    doc["_id"], str
                ):  # Solo si no es ya un string
                    doc["_id"] = str(doc["_id"])
            return {
                "status": "success",
                "operation": "aggregate",
                "data": result,
                "count": len(result),
            }

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Operación '{request.operation}' no soportada.",
            )

    except PyMongoError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en la operación de MongoDB: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado en la API: {e}",
        )


@app.get("/api/v1/dashboard", response_model=DashboardMetrics)
async def get_dashboard_data():
    """
    Retorna las métricas del dashboard calculadas a partir de la base de datos.
    """
    # 1. totalOrders
    total_orders_count = orders_collection.count_documents({})

    # 2. revenue (solo sumar órdenes con paid_status='paid')
    pipeline_total_revenue = [
        {"$match": {"paid_status": "paid"}},
        {
            "$group": {
                "_id": None,
                "totalRevenue": {"$sum": {"$toDouble": "$order_data.total_price"}},
            }
        },
    ]
    total_revenue_result = next(
        orders_collection.aggregate(pipeline_total_revenue), {"totalRevenue": 0.0}
    )
    total_revenue = total_revenue_result["totalRevenue"]

    # --- Nuevas Métricas ---

    # 3. unconfirmedOrders (Órdenes en "En Espera" (status=4))
    unconfirmed_orders_count = orders_collection.count_documents({"status": 4})

    # 4. confirmedOrders (Órdenes que no son "En Espera" (status=4) ni "Canceladas" (status=3))
    confirmed_orders_count = orders_collection.count_documents(
        {"status": {"$nin": [3, 4]}}
    )

    # 5. inTransitAndSoon (Órdenes en "En Tránsito" (status=0) y "Pronta Entrega" (status=1))
    in_transit_and_soon_count = orders_collection.count_documents(
        {"status": {"$in": [0, 1]}}
    )

    # 6. deliveredAndPaid (Órdenes entregadas (status=2) y pagadas (paid_status='paid'))
    delivered_and_paid_count = orders_collection.count_documents(
        {"status": 2, "paid_status": "paid"}
    )

    # 7. deliveredAndUnpaid (Órdenes entregadas (status=2) y NO pagadas)
    delivered_and_unpaid_count = orders_collection.count_documents(
        {
            "status": 2,
            "paid_status": {"$ne": "paid"},
        }
    )

    # 8. canceledOrders (Órdenes canceladas (status=3))
    canceled_orders_count = orders_collection.count_documents({"status": 3})

    # 9. averageOrderValue (Valor promedio de las órdenes pagadas)
    total_paid_orders_count = orders_collection.count_documents({"paid_status": "paid"})
    average_order_value = (
        (total_revenue / total_paid_orders_count)
        if total_paid_orders_count > 0
        else 0.0
    )

    # 10. deliveryRate
    delivered_orders_count = orders_collection.count_documents({"status": 2})
    total_eligible_for_delivery = orders_collection.count_documents(
        {"status": {"$ne": 3}}
    )
    delivery_rate = (
        (delivered_orders_count / total_eligible_for_delivery * 100)
        if total_eligible_for_delivery > 0
        else 0.0
    )

    return DashboardMetrics(
        totalOrders=total_orders_count,
        revenue=total_revenue,
        unconfirmedOrders=unconfirmed_orders_count,
        confirmedOrders=confirmed_orders_count,
        inTransitAndSoon=in_transit_and_soon_count,
        deliveredAndPaid=delivered_and_paid_count,
        deliveredAndUnpaid=delivered_and_unpaid_count,
        canceledOrders=canceled_orders_count,
        averageOrderValue=average_order_value,
        deliveryRate=delivery_rate,
    )


@app.get("/api/v1/orders", response_model=PaginatedResponse[MinimalOrder])
async def get_paginated_orders_from_db(
    page: int = Query(1, ge=1, description="Número de página"),
    page_size: int = Query(10, ge=1, le=100, description="Tamaño de página"),
    search: Optional[str] = Query(
        None, description="Buscar por orden, cliente, producto, email"
    ),
    payment_status: Optional[str] = Query(
        None, description="Filtrar por estado de pago"
    ),
    origin: Optional[str] = Query(None, description="Filtrar por origen de compra"),
):
    """
    Retorna una lista paginada de órdenes en formato MinimalOrder,
    permitiendo filtrado y búsqueda.
    """
    query = {}

    if search:
        search_normalized = unidecode(search).lower()
        search_clauses = []

        # Buscar por Shopify Order ID si es un número
        if search.isdigit():
            search_clauses.append({"shopify_order_id": int(search)})

        # Buscar por Kommo Order Number
        search_clauses.append(
            {"kommo_order_number": {"$regex": search_normalized, "$options": "i"}}
        )

        # Buscar por nombre completo del cliente
        search_clauses.append(
            {"basic_data.full_name": {"$regex": search_normalized, "$options": "i"}}
        )

        # Buscar por nombre de producto
        search_clauses.append(
            {"order_data.products.name": {"$regex": search_normalized, "$options": "i"}}
        )

        # Buscar por email del cliente
        search_clauses.append(
            {"basic_data.email": {"$regex": search_normalized, "$options": "i"}}
        )

        # Combinar las cláusulas de búsqueda con $or
        if search_clauses:
            query["$or"] = search_clauses

    if payment_status:
        query["paid_status"] = payment_status

    if origin:
        query["origen_compra"] = origin

    # Contar el total de documentos que coinciden con el filtro, sin paginación
    total_count = orders_collection.count_documents(query)

    skip_count = (page - 1) * page_size

    projection = {
        "shopify_order_id": 1,
        "kommo_order_number": 1,
        "fecha_compra": 1,
        "status": 1,
        "paid_status": 1,
        "order_data.total_price": 1,
        "basic_data.full_name": 1,
        "origen_compra": 1,
        "basic_data.phone": 1,
        "basic_data.email": 1,
        "_id": 0,
    }
    db_orders = list(
        orders_collection.find(query, projection)
        .sort("fecha_compra", -1)
        .skip(skip_count)
        .limit(page_size)
    )

    results = []
    for order in db_orders:
        total_price_str = order.get("order_data", {}).get("total_price", "0.0")
        try:
            total_price = float(total_price_str)
        except ValueError:
            total_price = 0.0

        results.append(
            MinimalOrder(
                id=order.get("shopify_order_id", -1),
                kommo_order_number=order.get(
                    "kommo_order_number", f"#{order.get('shopify_order_id', 'N/A')}"
                ),
                fecha_compra=order.get("fecha_compra", ""),
                status=order.get("status", -1),
                paid_status=order.get("paid_status", "unknown"),
                total=total_price,
                customerName=order.get("basic_data", {}).get(
                    "full_name", "Desconocido"
                ),
                origen_compra=order.get("origen_compra", "desconocido"),
                contactInfo={
                    "phone": order.get("basic_data", {}).get("phone", ""),
                    "email": order.get("basic_data", {}).get("email", ""),
                },
                # products=[] # No se incluyen productos en la vista MinimalOrder
            )
        )
    return PaginatedResponse(
        items=results, total_count=total_count, page=page, page_size=page_size
    )


@app.get("/api/v1/orders/recently", response_model=List[MinimalOrder])
async def get_recent_orders_from_db():
    """
    Retorna las 5 órdenes más recientes de la base de datos MongoDB.
    """
    includes = {
        "shopify_order_id": 1,
        "kommo_order_number": 1,
        "fecha_compra": 1,
        "status": 1,
        "paid_status": 1,
        "order_data.total_price": 1,
        "basic_data.full_name": 1,
        "origen_compra": 1,
        "basic_data.phone": 1,
        "basic_data.email": 1,
        "_id": 0,
    }
    data = list(orders_collection.find({}, includes).sort("fecha_compra", -1).limit(5))
    results = []
    for order in data:
        total_price_str = order.get("order_data", {}).get("total_price", "0.0")
        try:
            total_price = float(total_price_str)
        except ValueError:
            total_price = 0.0

        results.append(
            MinimalOrder(
                id=order.get("shopify_order_id", -1),
                kommo_order_number=order.get(
                    "kommo_order_number", f"#{order.get('shopify_order_id', 'N/A')}"
                ),
                fecha_compra=order.get("fecha_compra", ""),
                status=order.get("status", -1),
                paid_status=order.get("paid_status", "unknown"),
                total=total_price,
                customerName=order.get("basic_data", {}).get(
                    "full_name", "Desconocido"
                ),
                origen_compra=order.get("origen_compra", "desconocido"),
                contactInfo={
                    "phone": order.get("basic_data", {}).get("phone", ""),
                    "email": order.get("basic_data", {}).get("email", ""),
                },
                # products=[] # No se incluyen productos en la vista MinimalOrder
            )
        )
    return results


@app.get("/api/v1/orders/{order_id}", response_model=FullOrderDetails)
async def get_order_by_id(order_id: str):
    """
    Retorna todos los detalles de una orden específica desde la base de datos MongoDB.
    """
    try:
        order_id_int = int(order_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de orden inválido")

    order = orders_collection.find_one({"shopify_order_id": order_id_int})

    if not order:
        raise HTTPException(status_code=404, detail="Orden no encontrada")

    total_price_str = order.get("order_data", {}).get("total_price", "0.0")
    try:
        total_price = float(total_price_str)
    except ValueError:
        total_price = 0.0

    # --- INICIO DE LA CORRECCIÓN DE PIPELINE ---
    pipeline_val_from_db = order.get("pipeline", None)
    raw_pipeline_stages_list = []
    display_pipeline_str = "No especificado"

    if isinstance(pipeline_val_from_db, list) and pipeline_val_from_db:
        valid_stages = []
        for stage_item in pipeline_val_from_db:
            if isinstance(stage_item, dict):
                valid_stages.append(
                    {
                        "pipeline": stage_item.get("pipeline", "Desconocido"),
                        "status": stage_item.get("status", "Desconocido"),
                    }
                )
            elif isinstance(stage_item, str):
                try:
                    parsed_item = json.loads(stage_item.replace("'", '"'))
                    if isinstance(parsed_item, dict):
                        valid_stages.append(
                            {
                                "pipeline": parsed_item.get("pipeline", "Desconocido"),
                                "status": parsed_item.get("status", "Desconocido"),
                            }
                        )
                    else:
                        valid_stages.append(
                            {"pipeline": stage_item, "status": "Desconocido"}
                        )
                except json.JSONDecodeError:
                    valid_stages.append(
                        {"pipeline": stage_item, "status": "Desconocido"}
                    )

        if valid_stages:
            raw_pipeline_stages_list = valid_stages
            last_stage_dict = valid_stages[-1]
            pipeline_name = last_stage_dict.get("pipeline", "Desconocido")
            pipeline_status = last_stage_dict.get("status", "Desconocido")
            display_pipeline_str = f"{pipeline_name} - {pipeline_status}"
        else:
            display_pipeline_str = "No especificado"
            raw_pipeline_stages_list = []
    elif (
        isinstance(pipeline_val_from_db, (str, int, float))
        and str(pipeline_val_from_db).strip()
    ):
        display_pipeline_str = str(pipeline_val_from_db)
        raw_pipeline_stages_list = [
            {"pipeline": display_pipeline_str, "status": "Única etapa"}
        ]
    else:
        display_pipeline_str = "No especificado"
        raw_pipeline_stages_list = []
    # --- FIN DE LA CORRECCIÓN DE PIPELINE ---

    products_for_frontend = []
    for p in order.get("order_data", {}).get("products", []):
        product_images = []
        if p.get("image_url"):
            product_images.append(p["image_url"])

        products_for_frontend.append(
            {
                "name": p.get("name"),
                "price": p.get("price"),
                "quantity": p.get("quantity", 1),
                "producto_id": p.get("producto_id"),
                "variant_title": p.get("variant_title", "Estándar"),
                "sku": p.get("sku", None),
                "product_urls": product_images,
            }
        )

    shipping_lines = order.get("shipping_lines", [])
    if not isinstance(shipping_lines, list):
        shipping_lines = []

    data = {
        "id": order.get("shopify_order_id", -1),
        "kommo_order_number": order.get(
            "kommo_order_number", f"#{order.get('shopify_order_id', 'N/A')}"
        ),
        "kommoLeadId": order.get("kommo_lead_id", None),
        "kommoContactId": order.get("kommo_contact_id", None),
        "fecha_compra": order.get("fecha_compra", ""),
        "fecha_confirmacion": order.get("fecha_confirmacion", None),
        "fecha_entrega": order.get("fecha_entrega", None),
        "fecha_pago": order.get("fecha_pago", None),
        "tipo_pago": order.get("tipo_pago", None),
        "tracking_code": order.get("tracking_url", None),
        "paqueteria": order.get("paqueteria", None),
        "origen_compra": order.get("origen_compra", None),
        "origen_contacto": order.get("origen_contacto", None),
        "confirmador_name": order.get("confirmador_name", None),
        "cobrador_name": order.get("cobrador_name", None),
        "NOTA": order.get("NOTA", None),
        "nota_n8n": order.get("nota_n8n", None),
        "OBSERVACIONES": order.get("OBSERVACIONES", None),
        "pipeline": display_pipeline_str,
        "raw_pipeline_stages": raw_pipeline_stages_list,
        "status": order.get("status", -1),
        "paidStatus": order.get("paid_status", "unknown"),
        "user_type": order.get("user_type", "cliente"),
        "order_type": order.get("order_type", "Orden de Compra"),
        "profs_url": order.get("profs_url", []),
        "customerName": order.get("basic_data", {}).get("full_name", "Desconocido"),
        "total": total_price,
        "date": order.get("fecha_compra", ""),
        "products": products_for_frontend,
        "basic_data": order.get("basic_data", {}),
        "order_data": order.get("order_data", {}),
        "shipping_lines": shipping_lines,
        "billing_address": order.get("billing_address", {}),
        "contactInfo": {
            "phone": order.get("basic_data", {}).get("phone", ""),
            "email": order.get("basic_data", {}).get("email", ""),
        },
    }

    return data


@app.get("/api/v1/clients", response_model=PaginatedResponse[Client])
async def get_paginated_clients_from_db(
    page: int = Query(1, ge=1, description="Número de página"),
    page_size: int = Query(10, ge=1, le=100, description="Tamaño de página"),
    search: Optional[str] = Query(
        None, description="Buscar por nombre, correo, teléfono"
    ),
    client_type: Optional[str] = Query(
        None, description="Filtrar por tipo de cliente (client, repeat)"
    ),
):
    """
    Retorna una lista paginada de clientes con un resumen de sus órdenes.
    """
    # Consulta base para la agregación
    base_pipeline = [
        {"$match": {"basic_data.id": {"$exists": True, "$ne": None}}},
    ]

    # Añadir filtros de búsqueda y tipo de cliente al pipeline
    if search:
        search_normalized = unidecode(search).lower()
        base_pipeline.append(
            {
                "$match": {
                    "$or": [
                        {
                            "basic_data.full_name": {
                                "$regex": search_normalized,
                                "$options": "i",
                            }
                        },
                        {
                            "basic_data.email": {
                                "$regex": search_normalized,
                                "$options": "i",
                            }
                        },
                        {
                            "basic_data.phone": {
                                "$regex": search_normalized,
                                "$options": "i",
                            }
                        },
                    ]
                }
            }
        )

    # Agrupación y cálculo de métricas por cliente
    group_stage = {
        "$group": {
            "_id": "$basic_data.id",
            "name": {"$first": "$basic_data.full_name"},
            "email": {"$first": "$basic_data.email"},
            "phone": {"$first": "$basic_data.phone"},
            "totalSpent": {"$sum": {"$toDouble": "$order_data.total_price"}},
            "orderCount": {"$sum": 1},
            "lastOrderDate": {"$max": "$fecha_compra"},
            "orders": {
                "$push": {
                    "id": "$shopify_order_id",
                    "kommo_order_number": "$kommo_order_number",
                    "fecha_compra": "$fecha_compra",
                    "total": {"$toDouble": "$order_data.total_price"},
                    "status": "$status",
                    "paid_status": "$paid_status",
                }
            },
        }
    }
    base_pipeline.append(group_stage)

    # Añadir la lógica de userType (debe ir después del $group)
    project_user_type_stage = {
        "$project": {
            "id": "$_id",
            "name": 1,
            "email": 1,
            "phone": 1,
            "totalSpent": 1,
            "orderCount": 1,
            "lastOrderDate": 1,
            "orders": 1,
            "userType": {
                "$cond": {
                    "if": {"$gt": ["$orderCount", 1]},
                    "then": "repeat",
                    "else": "client",
                }
            },
            "_id": 0,
        }
    }
    base_pipeline.append(project_user_type_stage)

    # Aplicar filtro por tipo de cliente si se proporciona
    if client_type:
        base_pipeline.append({"$match": {"userType": client_type}})

    # Obtener el total de clientes *después* de aplicar todos los filtros
    total_count_pipeline = base_pipeline + [{"$count": "total_clients"}]
    total_clients_result = next(
        orders_collection.aggregate(total_count_pipeline), {"total_clients": 0}
    )
    total_count = total_clients_result["total_clients"]

    # Ordenar y aplicar paginación
    final_pipeline = base_pipeline + [
        {"$sort": {"lastOrderDate": -1}},
        {"$skip": (page - 1) * page_size},
        {"$limit": page_size},
    ]

    db_clients = list(orders_collection.aggregate(final_pipeline))

    results = []
    for client_doc in db_clients:
        client_doc["totalSpent"] = float(client_doc.get("totalSpent", 0.0))
        processed_orders = []
        for order_summary in client_doc.get("orders", []):
            order_summary["total"] = float(order_summary.get("total", 0.0))
            processed_orders.append(ClientOrderSummary(**order_summary))
        client_doc["orders"] = processed_orders
        results.append(Client(**client_doc))

    return PaginatedResponse(
        items=results, total_count=total_count, page=page, page_size=page_size
    )


@app.get("/api/v1/activity", response_model=List[ActivityLog])
async def get_activity_feed():
    """
    Actualmente devuelve una lista vacía o podrías integrarlo con una colección
    de logs en MongoDB si existiera.
    """
    return []


@app.get("/api/v1/integrations/shopify", response_model=IntegrationStats)
async def get_shopify_data():
    """
    Mantiene la información de configuración estática, no se considera 'mock data' de negocio.
    """
    return IntegrationStats(
        status="Activo",
        connected_account="ofertasenmexico.com",
        stats={"products_synced": 156, "last_sync": "Hace 5 min", "active_webhooks": 4},
    )


@app.get("/api/v1/integrations/kommo", response_model=IntegrationStats)
async def get_kommo_data():
    """
    Mantiene la información de configuración estática, no se considera 'mock data' de negocio.
    """
    return IntegrationStats(
        status="Activo",
        connected_account="cuenta.kommo.com",
        stats={"leads_total": 2847, "pipelines": 3, "stages": 12, "users": 8},
    )


@app.get("/api/v1/notifications", response_model=List[Notification])
async def get_notifications():
    """
    Devuelve una lista vacía. Podrías integrarlo con una colección 'notifications' de MongoDB.
    """
    return []


@app.get("/api/v1/blacklist", response_model=List[BlacklistUser])
async def get_blacklist():
    """
    Devuelve una lista vacía. Podrías integrarlo con una colección 'blacklist_users' de MongoDB.
    """
    return []


@app.get("/api/v1/debtors", response_model=List[Debtor])
async def get_debtors():
    """
    Devuelve una lista vacía. Podrías integrarlo con una colección 'debtors' de MongoDB.
    """
    return []


@app.get("/api/v1/analytics/charts", response_model=ChartData)
async def get_chart_data():
    """
    Genera datos básicos para gráficos a partir de la base de datos.
    """
    top_products_pipeline = [
        {"$unwind": "$order_data.products"},
        {
            "$group": {
                "_id": "$order_data.products.name",
                "total_sales": {"$sum": "$order_data.products.quantity"},
                "total_revenue": {
                    "$sum": {
                        "$multiply": [
                            {"$toDouble": "$order_data.products.quantity"},
                            {"$toDouble": "$order_data.products.price"},
                        ]
                    }
                },
            }
        },
        {"$sort": {"total_revenue": -1}},
        {"$limit": 3},
        {
            "$project": {
                "_id": 0,
                "name": "$_id",
                "sales": "$total_sales",
                "revenue": "$total_revenue",
                "icon": "fa-box",
                "color": "gray-500",
            }
        },
    ]
    top_products_results = list(orders_collection.aggregate(top_products_pipeline))

    for product in top_products_results:
        product["revenue"] = float(product["revenue"])

    return ChartData(
        revenue_labels=[f"Sem {i}" for i in range(1, 9)],  # Datos dummy
        revenue_current=[0.0] * 8,  # Datos dummy
        revenue_previous=[0.0] * 8,  # Datos dummy
        conversions_labels=[
            "Lun",
            "Mar",
            "Mié",
            "Jue",
            "Vie",
            "Sáb",
            "Dom",
        ],  # Datos dummy
        conversions_data=[0] * 7,  # Datos dummy
        top_products=top_products_results,
    )


# --- LOGIN ---
@app.post("/api/v1/login", response_model=UserResponse)
async def login_for_access_token(credentials: LoginRequest):
    user_doc = users_collection.find_one({"email": credentials.email})
    if not user_doc or user_doc["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    return UserResponse(
        email=user_doc["email"], name=user_doc["name"], role=user_doc["role"]
    )


@app.post("/api/v1/orders/search", response_model=List[MinimalOrder])
async def search_orders_from_db(filters: SearchFilters):
    """
    Endpoint de búsqueda avanzada que recibe un objeto JSON con los filtros en el cuerpo de la solicitud.
    """
    query = {}

    # --- Filtros de texto normalizados (sin tildes, en minúsculas) ---
    if filters.order_number:
        query["kommo_order_number"] = {
            "$regex": unidecode(filters.order_number),
            "$options": "i",
        }

    if filters.partialName:
        part_normalized = unidecode(filters.partialName).lower()
        query["$or"] = [
            {
                "basic_data.full_name": {
                    "$regex": part_normalized,
                    "$options": "i",
                }
            },
            {
                "basic_data.email": {
                    "$regex": part_normalized,
                    "$options": "i",
                }
            },
        ]

    if filters.street:
        query["basic_data.address"] = {
            "$regex": unidecode(filters.street),
            "$options": "i",
        }
    if filters.city:
        query["basic_data.ciudad"] = {
            "$regex": unidecode(filters.city),
            "$options": "i",
        }
    if filters.product:
        query["order_data.products.name"] = {
            "$regex": unidecode(filters.product),
            "$options": "i",
        }
    if filters.origin:
        query["origen_compra"] = {"$regex": unidecode(filters.origin), "$options": "i"}

    # --- Filtros de comparación exacta (o limpieza) ---
    if filters.phone:
        clean_phone = "".join(filter(str.isdigit, filters.phone))
        query["basic_data.phone"] = {"$regex": clean_phone}
    if filters.email:
        query["basic_data.email"] = {
            "$regex": unidecode(filters.email),
            "$options": "i",
        }

    # --- Filtros que no requieren unidecode ---
    if filters.collectorName:
        query["cobrador_name"] = {
            "$regex": unidecode(filters.collectorName),
            "$options": "i",
        }
    if filters.confirmerName:
        query["confirmador_name"] = {
            "$regex": unidecode(filters.confirmerName),
            "$options": "i",
        }
    if filters.shopifyId:
        query["shopify_order_id"] = filters.shopifyId
    if filters.customerId:
        query["basic_data.id"] = filters.customerId
    if filters.postalCode:
        query["basic_data.postal_code"] = filters.postalCode
    if isinstance(filters.status, int):
        query["status"] = filters.status
    if filters.paidStatus:
        query["paid_status"] = filters.paidStatus

    # --- Filtros de fecha por rango ---
    # Los campos de fecha en MongoDB deben ser consistentes con el formato esperado (YYYY-MM-DD)
    # y si están como string, se compararán como string.

    # Fecha de Compra
    if filters.dateFrom or filters.dateTo:
        date_query = {}
        if filters.dateFrom:
            converted_date_from = convert_date_format(filters.dateFrom)
            if converted_date_from:
                date_query["$gte"] = converted_date_from
        if filters.dateTo:
            converted_date_to = convert_date_format(filters.dateTo)
            if converted_date_to:
                date_query["$lte"] = converted_date_to
        if date_query:
            query["fecha_compra"] = date_query
    elif filters.date:
        converted_date = convert_date_format(filters.date)
        if converted_date:
            query["fecha_compra"] = converted_date

    # Fecha de Confirmación
    if filters.dateFromConfirmacion or filters.dateToConfirmacion:
        date_query = {}
        if filters.dateFromConfirmacion:
            converted_date_from = convert_date_format(filters.dateFromConfirmacion)
            if converted_date_from:
                date_query["$gte"] = converted_date_from
        if filters.dateToConfirmacion:
            converted_date_to = convert_date_format(filters.dateToConfirmacion)
            if converted_date_to:
                date_query["$lte"] = converted_date_to
        if date_query:
            query["fecha_confirmacion"] = date_query
    elif filters.dateConfirmacion:
        converted_date = convert_date_format(filters.dateConfirmacion)
        if converted_date:
            query["fecha_confirmacion"] = converted_date

    # Fecha de Entrega
    if filters.dateFromEntrega or filters.dateToEntrega:
        date_query = {}
        if filters.dateFromEntrega:
            converted_date_from = convert_date_format(filters.dateFromEntrega)
            if converted_date_from:
                date_query["$gte"] = converted_date_from
        if filters.dateToEntrega:
            converted_date_to = convert_date_format(filters.dateToEntrega)
            if converted_date_to:
                date_query["$lte"] = converted_date_to
        if date_query:
            query["fecha_entrega"] = date_query
    elif filters.dateEntrega:
        converted_date = convert_date_format(filters.dateEntrega)
        if converted_date:
            query["fecha_entrega"] = converted_date

    # Fecha de Pago
    if filters.dateFromPago or filters.dateToPago:
        date_query = {}
        if filters.dateFromPago:
            converted_date_from = convert_date_format(filters.dateFromPago)
            if converted_date_from:
                date_query["$gte"] = converted_date_from
        if filters.dateToPago:
            converted_date_to = convert_date_format(filters.dateToPago)
            if converted_date_to:
                date_query["$lte"] = converted_date_to
        if date_query:
            query["fecha_pago"] = date_query
    elif filters.datePago:
        converted_date = convert_date_format(filters.datePago)
        if converted_date:
            query["fecha_pago"] = converted_date

    # --- Proyección y Ejecución de la consulta ---
    # Para el search results section, el frontend espera MinimalOrder
    projection = {
        "_id": 0,
        "shopify_order_id": 1,
        "kommo_order_number": 1,
        "fecha_compra": 1,
        "status": 1,
        "paid_status": 1,
        "order_data.total_price": 1,
        "basic_data.full_name": 1,
        "origen_compra": 1,
        "basic_data.phone": 1,
        "basic_data.email": 1,
    }

    try:
        # Limita a 1000 resultados para evitar sobrecargar el frontend con búsquedas muy amplias
        db_orders_cursor = orders_collection.find(query, projection).limit(1000)
        db_orders = list(db_orders_cursor)

        results = []
        for order in db_orders:
            total_price_str = order.get("order_data", {}).get("total_price", "0.0")
            try:
                total_price = float(total_price_str)
            except ValueError:
                total_price = 0.0

            results.append(
                MinimalOrder(
                    id=order.get("shopify_order_id", -1),
                    kommo_order_number=order.get(
                        "kommo_order_number", f"#{order.get('shopify_order_id', 'N/A')}"
                    ),
                    fecha_compra=order.get("fecha_compra", ""),
                    status=order.get("status", -1),
                    paid_status=order.get("paid_status", "unknown"),
                    total=total_price,
                    customerName=order.get("basic_data", {}).get(
                        "full_name", "Desconocido"
                    ),
                    origen_compra=order.get("origen_compra", "desconocido"),
                    contactInfo={
                        "phone": order.get("basic_data", {}).get("phone", ""),
                        "email": order.get("basic_data", {}).get("email", ""),
                    },
                    # products=[] # No se incluyen productos en la vista MinimalOrder
                )
            )

        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ocurrió un error al procesar la búsqueda en la base de datos: {str(e)}",
        )


@app.put("/api/v1/orders/{order_id}/flags")
async def update_order_flags(order_id: int, flags: OrderFlagsUpdate):
    """
    Actualiza los flags 'is_deudor' y 'is_blacklist' de una orden específica.
    """
    try:
        update_data = {}
        if flags.is_deudor is not None:
            update_data["is_deudor"] = flags.is_deudor
        if flags.is_blacklist is not None:
            update_data["is_blacklist"] = flags.is_blacklist

        if not update_data:
            raise HTTPException(
                status_code=400, detail="No se proporcionaron campos para actualizar."
            )

        update_result = orders_collection.update_one(
            {"shopify_order_id": order_id}, {"$set": update_data}
        )

        if update_result.modified_count == 0:
            # Podría ser que la orden no existe o que los valores ya eran los mismos
            # En un entorno real, podrías verificar primero si la orden existe
            # para diferenciar entre "no encontrada" y "no se realizaron cambios".
            existing_order = orders_collection.find_one({"shopify_order_id": order_id})
            if not existing_order:
                raise HTTPException(status_code=404, detail="Orden no encontrada.")
            else:
                return {
                    "message": "No se realizaron cambios, los valores ya eran los mismos."
                }

        return {"message": "Estado de la orden actualizado correctamente."}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al actualizar el estado: {str(e)}"
        )


@app.post("/api/v1/create_lead")
async def create_lead_endpoint(lead_data: CreateLeadRequest):
    print(f"Order ID: {lead_data.order_id}")
    print(f"Kommo Order Number: {lead_data.kommo_order_number}")
    print(f"Nota del Lead: {lead_data.nota}")
    print(f"Email del Usuario: {lead_data.email}")
    print("---------------------------------------")
    """
    # Ejemplo de cómo podrías interactuar con Kommo o guardar en DB
    # Aquí puedes añadir la lógica real para crear un lead en Kommo o en tu DB de leads.
    # Por ejemplo:
    # orders_collection.update_one(
    #    {"shopify_order_id": lead_data.order_id},
    #    {"$set": {"lead_created": True, "lead_notes": lead_data.nota, "lead_creator_email": lead_data.email}}
    # )
    # raise HTTPException(status_code=401, detail="No tienes permiso para realizar esta acción.") # Ejemplo de error
    """
    return {
        "status": 200,
        "message": "Tarea de creación de lead iniciada.",
    }
    # return {
    #     "status": 401,
    #     "message": "No tienes permiso para realizar esta acción.",
    # }


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Sirve el archivo index.html principal."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html no encontrado.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
