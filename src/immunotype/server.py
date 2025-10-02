"""
Production server for immunotype web application.

This module provides multiple ways to serve the immunotype web interface:
1. Built-in Gradio server (recommended for most use cases)
2. ASGI compatibility for uvicorn, gunicorn, and container deployments

Usage:
    # Development/production server with Gradio
    python -m immunotype.server
    python -c "from immunotype.server import launch; launch()"

    # ASGI server for container deployments (experimental)
    uvicorn immunotype.server:asgi_app --host 0.0.0.0 --port 8000
"""


def create_app():
    """Create the Gradio application."""
    from .web import create_interface

    return create_interface()


def launch(host="0.0.0.0", port=8000, share=False, **kwargs):
    """
    Launch the immunotype web server using Gradio's built-in server.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
        share: Create a public Gradio link (default: False)
        **kwargs: Additional arguments passed to Gradio launch()
    """
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share, show_error=True, **kwargs)


# ASGI app for container deployments (experimental)
# Note: Some Gradio versions have ASGI compatibility issues
def create_asgi_app():
    """Create ASGI-compatible app (experimental - use launch() for production)."""
    return create_app()


# Lazy ASGI app creation to avoid import-time issues
_asgi_app = None


def get_asgi_app():
    """Get or create the ASGI app."""
    global _asgi_app
    if _asgi_app is None:
        _asgi_app = create_asgi_app()
    return _asgi_app


# Export ASGI app for uvicorn
asgi_app = get_asgi_app

# For backwards compatibility
app = get_asgi_app
main = launch

if __name__ == "__main__":
    import rich_click as click

    @click.command()
    @click.option("--host", default="0.0.0.0", help="Host to bind to", show_default=True)
    @click.option("--port", type=int, default=8000, help="Port to bind to", show_default=True)
    @click.option("--share", is_flag=True, help="Create public Gradio link for sharing")
    def main_cli(host, port, share):
        """Launch the immunotype web server."""
        launch(host=host, port=port, share=share)

    main_cli()
