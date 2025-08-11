from prometheus_fastapi_instrumentator import Instrumentator


instrumentator = Instrumentator()

def register_metrics(app) -> None:
    instrumentator.instrument(app).expose(app)


