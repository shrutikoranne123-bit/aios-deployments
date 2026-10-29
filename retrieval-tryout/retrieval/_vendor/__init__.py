"""
Vendored copies of the few `services.*` modules the retrieval service depends
on, so the `retrieval/` package is fully self-contained and deployable on its
own (no dependency on the top-level `services/` tree).

Modules here are byte-for-byte equivalents of their `services/` counterparts,
chosen because they are lightweight and dependency-free (pure stdlib / pydantic):

  api_models      <- services.api_upload        (auth + pagination models; HARD dep)
  token_tracker   <- services.token_tracker     (best-effort token accounting)
  version_service <- services.version_service   (available-versions lookup)
  json_extractor  <- services.extraction.extractors.json_extractor (api tool kind)

The retrieval modules import these with a `try: services.* / except: _vendor.*`
pattern, so the SAME code runs both inside the monorepo (using services/) and
in the standalone retrieval image (using this vendor package).
"""
