# Image Translation

## Runtime configuration

All deployment-specific settings are read through
`image_translation.config.get_settings()`. Application modules do not read
environment variables directly.

Create the local configuration before starting the service:

```bash
cp .env.example .env
```

Then update the upstream LLM/VLM URLs, OSS URL, font file, and test directories
in `.env`. Process environment variables override values from the file. To load
a different file, set `IMAGE_TRANSLATION_ENV_FILE` to its path.

Terms that should remain untranslated are maintained in
`configs/no_translate_terms.json`, grouped by category. Its location is set by
`NO_TRANSLATE_TERMS_FILE`; changing the terms takes effect after a restart.

LLM/VLM prompt templates use structured JSON under `configs/prompts/`:
`vision.json` contains recognition prompts, and `translations.json` contains
all `translate_*` prompts. Set `PROMPTS_DIR` to use a different directory;
changes take effect after a restart.

Configuration is validated on startup. Missing required URLs, endpoints,
prompts, or paths raise a `ConfigurationError` with the exact variable name.

Start the service from the repository root:

```bash
PYTHONPATH=src python server.py
```

FastAPI route paths are bound at application startup, so route configuration
changes require a restart. Uvicorn reload can be enabled locally with
`SERVER_RELOAD=true`.
