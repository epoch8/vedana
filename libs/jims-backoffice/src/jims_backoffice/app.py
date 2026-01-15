from fastapi import FastAPI, status
from fastapi.responses import HTMLResponse
from fastui import prebuilt_html


def set_routes(
    app: FastAPI,
) -> FastAPI:
    TITLE = "Backoffice"

    @app.get("/healthz", status_code=status.HTTP_200_OK)
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/{path:path}")
    async def html_landing():
        """Simple HTML page which serves the React app, comes last as it matches all paths."""
        return HTMLResponse(prebuilt_html(title=TITLE))

    return app


def get_application() -> FastAPI:
    from jims_backoffice.main_app import application
    from jims_backoffice.routes import event, home  # noqa

    set_routes(application)
    return application


app = get_application()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
