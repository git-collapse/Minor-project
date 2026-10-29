# Vercel Serverless Entrypoint
# Exports top-level 'app', 'handler', and 'application' variables expected by Vercel
from api.index import app, handler, application

__all__ = ["app", "handler", "application"]
