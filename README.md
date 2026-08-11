# Image Translation

## Runtime configuration

All deployment-specific settings are read through
`image_translation.config.get_settings()`. Application modules do not read
environment variables directly.

Create the local configuration before starting the service:

```bash
cp .env.example .env
```

Install the project and development test dependencies in the active virtual
environment:

```bash
python -m pip install -e '.[dev]'
```

For a CPU-only OCR environment, include the optional PaddlePaddle runtime:

```bash
python -m pip install -e '.[cpu,dev]'
```

GPU deployments should install the `paddlepaddle-gpu` build matching the
server's CUDA version separately instead of using the `cpu` extra.

Select the OCR device explicitly in `.env`:

```dotenv
# Safe default, including environments with the CPU PaddlePaddle wheel
OCR_DEVICE=cpu
OCR_GPU_ID=0
```

Set `OCR_DEVICE=gpu` on a GPU deployment. `OCR_GPU_ID` is zero-based and is
used by both PaddleOCR engines. Startup fails immediately if the installed
PaddlePaddle runtime has no CUDA support or the requested GPU does not exist;
it never silently falls back to CPU. The environment variable selects a device
but does not replace the installed runtime, so GPU mode still requires a
CUDA-compatible `paddlepaddle-gpu` package.

Use a dedicated environment for this service. The supported PaddleOCR 2.x
stack pins NumPy below 2 and OpenCV below 4.12; current vLLM environments may
require newer, incompatible OpenAI/OpenCV packages. Installing `.[cpu,dev]`
inside a vLLM environment can also replace an existing PaddlePaddle 3.x build.
For example:

```bash
conda create -n image-translation python=3.10 -y
conda activate image-translation
python -m pip install -e '.[cpu,dev]'
python -m pip check
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
python server.py
```

## MCP access

The same process exposes the legacy REST endpoint and an MCP Streamable HTTP
endpoint. MCP is enabled by default at:

```text
http://127.0.0.1:8000/mcp
```

The MCP server currently exposes one structured tool:

```text
translate_image_from_oss(bucket, image_key, language="en_zh",
                         save_bucket=None, output_key=None, segment=False)
```

The tool reads the source image from the configured object-storage service,
writes the translated image back to object storage, and returns the source and
translated object locations together with translated text regions. It accepts
object references rather than base64 image data so large images do not enter an
agent's context.

Test the endpoint with MCP Inspector after the service is running:

```bash
npx -y @modelcontextprotocol/inspector
```

Then connect the Inspector to `http://127.0.0.1:8000/mcp`.

`MCP_ALLOWED_HOSTS` and `MCP_ALLOWED_ORIGINS` intentionally default to local
clients. Before deploying behind a hostname, replace them with the exact public
host and browser origins. Do not expose this compute-intensive tool publicly
without TLS, authentication, rate limiting, and request quotas. The first
version does not configure an OAuth provider; keep it on a trusted network until
deployment authentication is added.

The OCR runtime is process-local and loads PaddleOCR once. Run one Uvicorn
worker per service instance; using multiple Uvicorn workers loads a separate OCR
runtime in every worker.

Run the focused unit-test suite with:

```bash
python -m pytest
```

Pytest uses importlib import mode so duplicate test filenames in legacy copied
test directories cannot shadow one another. Only `test_*.py` files are
collected automatically, and legacy copied `unit_tests`, `integration`, and
`integration_tests` directories are excluded from the default suite. The
`tests/integration/*_test.py` files are manual scripts that may initialize
PaddleOCR or call live upstream services and should be run explicitly when
those dependencies are available.

FastAPI route paths are bound at application startup, so route configuration
changes require a restart. Uvicorn reload can be enabled locally with
`SERVER_RELOAD=true`.
# poll scm test
