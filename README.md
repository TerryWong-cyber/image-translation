# Image Translation

## Synchronous translation APIs

Both REST APIs keep the HTTP request open until translation finishes. There is
no task submission or status polling API.

### URL input → stored result URL (public REST contract)

`POST /llm/image-translate` accepts JSON containing a **complete HTTP(S) URL**:

```bash
curl -X POST http://127.0.0.1:8000/llm/image-translate \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"https://example.com/image.png","language":"en_zh","segment":false}'
```

```json
{
  "status": "success",
  "request_id": "generated-uuid",
  "source_url": "https://example.com/image.png",
  "translated_url": "https://files.example.com/file/image-translations/translations/generated-uuid.png",
  "data": [],
  "duration_ms": 1234
}
```

The service downloads the image, translates it, and uploads the result to
`OSS_DEFAULT_BUCKET`, using a unique key per request. The response is sent only
after upload succeeds. The source URL may be a public image, a signed URL, or a
complete file URL on the configured OSS service. No bucket/key fields are
accepted by this REST API; old bucket/key REST requests must migrate to complete
URLs. MCP follows the same URL contract; no OSS-reference compatibility tool
is exposed.

Pre-create `OSS_DEFAULT_BUCKET` with write access. Set `OSS_PUBLIC_BASE_URL` to
the externally reachable file-service base URL (including any proxy base path).
The configured `OSS_FILE_ENDPOINT` is appended to it with URL-escaped bucket/key
components. If unset, it falls back to `OSS_BASE_URL`, which is only suitable when
that address is also reachable by callers. Result links use the file service's
existing access policy; this service does not create signed output links or
make private buckets public. Configure a caller-accessible download route and
any result retention/lifecycle policy on the OSS service.

Downloads have a total deadline, a byte limit, and a redirect limit. Each public
URL/redirect resolves only to global IP addresses; connections are pinned to the
checked IP while retaining HTTP Host and TLS SNI. Private addresses are allowed
only for the configured `OSS_BASE_URL` origin and file-route prefix. Signed input
queries are preserved, and no environment proxy or authorization header is
forwarded. HTTP-level compressed responses are rejected to keep the byte bound
predictable. Private OSS requiring additional credentials is not supported;
provide a signed input URL or an accessible OSS file route.

### Raw image input → raw image output (internal only)

`POST /internal/image-translate?language=en_zh&segment=false` accepts an image as
the raw request body and returns image bytes with the appropriate `Content-Type`.
It neither downloads nor stores images. It is disabled by default, absent from
OpenAPI, and registered only when the following settings are supplied:

```dotenv
API_BINARY_ENABLED=true
API_BINARY_TOKEN=replace-with-a-strong-service-token
```

Enabling it without a token fails configuration validation. Restrict this route
to internal traffic at the gateway as well.

```bash
curl -X POST 'http://127.0.0.1:8000/internal/image-translate?language=en_zh' \
  -H "Authorization: Bearer $API_BINARY_TOKEN" \
  -H 'Content-Type: image/png' \
  --data-binary @input.png \
  --output translated.png
```

JPEG, PNG, WebP, BMP and TIFF output encoding follows the existing format-aware
encoding behavior. Both paths validate image decoding and pixel count before
OCR. Defaults are 20 MiB of input bytes, 40 million pixels, 15 seconds for source
downloads, and 3 redirects. Configure `API_MAX_IMAGE_BYTES`,
`API_MAX_IMAGE_PIXELS`, `API_DOWNLOAD_TIMEOUT_SECONDS`, and `API_MAX_REDIRECTS` as
needed. These are download/input bounds, not an end-to-end translation deadline;
OCR and model upstreams retain their existing timeout settings.

Invalid request schemas return 422; invalid images/unsafe URLs return 400;
oversized images return 413; binary authentication failures return 401;
unsupported binary request media types return 415; download/upstream HTTP errors
return 502; download/OCR timeouts return 504; other processing/storage failures
return 500. Error bodies are JSON even for the binary endpoint.

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

GPU deployments should install PaddlePaddle GPU separately instead of using
the `cpu` extra. For the Blackwell server with a CUDA 13-capable driver:

```bash
python -m pip install paddlepaddle-gpu==3.3.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
python -m pip install -e '.[gpu,dev]'
python -m pip check
```

Select the OCR device explicitly in `.env`:

```dotenv
# Safe default, including environments with the CPU PaddlePaddle wheel
OCR_DEVICE=cpu
OCR_GPU_ID=0
OCR_VERSION=PP-OCRv4
```

Set `OCR_DEVICE=gpu` on a GPU deployment. `OCR_GPU_ID` is the zero-based index
among the GPUs visible to the process. Startup fails immediately if the
installed PaddlePaddle runtime has no CUDA support or the requested GPU does
not exist; it never silently falls back to CPU. The environment variable
selects a device but does not replace the installed runtime, so GPU mode still
requires a CUDA-compatible `paddlepaddle-gpu` package.

The service uses PaddleOCR 3.x and converts its structured results back into
the existing internal OCR contract, while the internal OSS-reference MCP contract stays compatible. `OCR_VERSION=PP-OCRv4` preserves the previous PaddleOCR 2.9 model
baseline; it can be changed independently after image-quality regression tests.

Use a dedicated environment for this service. PaddleOCR 3.x installs PaddleX
and its matching OpenCV contrib runtime; current vLLM environments may require
different OpenAI/OpenCV packages. Installing OCR dependencies inside a vLLM
environment can therefore replace packages used by the model server.
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

## Production GPU container

`Dockerfile.gpu` is the production image. It installs PaddlePaddle GPU 3.3.0
for CUDA 13, installs the application without development dependencies, and
downloads these fixed models while the image is built:

```text
PP-LCNet_x1_0_textline_ori
PP-OCRv4_mobile_det
en_PP-OCRv4_mobile_rec
```

The models are stored inside the image under
`/opt/paddlex-cache/official_models`. The build also writes
`/opt/paddlex-cache/models.sha256.json`, containing the size and SHA-256 digest
of every model file. Production startup does not download models.

Prepare the production configuration:

```bash
cp .env.production.example .env.production
```

Update the LLM/VLM and OSS URLs, public MCP hostname/origins, and choose the
physical GPU exposed to the container:

```bash
export NVIDIA_VISIBLE_DEVICES=0
```

Build and start locally:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f image-translation
```

Verify liveness and OCR readiness:

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
```

For repeatable deployment, build the image once in CI, push it to a private
registry, and let production hosts pull that immutable image:

```bash
export IMAGE_TRANSLATION_IMAGE=registry.example.com/image-translation:0.1.0-gpu-cu130
docker compose build
docker compose push

# On a production host with .env.production already configured:
docker compose pull
docker compose up -d
```

The host needs a compatible NVIDIA driver, Docker Engine, Docker Compose, and
NVIDIA Container Toolkit. It does not need Conda, PaddleOCR, PaddlePaddle, or
the model files installed separately.

`OCR_REQUIRE_LOCAL_MODELS=true` makes the container fail before OCR starts if
one of the configured model directories is missing. The application passes
the following explicit paths to PaddleOCR instead of relying on a user-home
cache:

```dotenv
OCR_TEXTLINE_MODEL_DIR=/opt/paddlex-cache/official_models/PP-LCNet_x1_0_textline_ori
OCR_DETECTION_MODEL_DIR=/opt/paddlex-cache/official_models/PP-OCRv4_mobile_det
OCR_RECOGNITION_MODEL_DIR=/opt/paddlex-cache/official_models/en_PP-OCRv4_mobile_rec
```

The model artifacts are small compared with the GPU runtime. Most image size
comes from PaddlePaddle GPU and its CUDA, cuDNN, cuBLAS, and cuFFT libraries.
Use registry and Docker layer caching rather than installing these packages on
every production start.

The existing `Dockerfile` remains a lightweight development shell. Use
`Dockerfile.gpu` or `docker-compose.yml` for production.

## MCP access

The same process exposes the URL-based REST endpoint and an MCP Streamable HTTP
endpoint. MCP is enabled by default at:

```text
http://127.0.0.1:8000/mcp
```

The MCP server currently exposes one structured tool:

```text
translate_image_from_url(image_url, language, segment=False, dry_run=False)
```

The tool uses the same URL download, byte translation, and default-bucket
persistence path as REST. It returns complete source and result URLs,
`request_id`, `duration_ms`, and translated text regions. Agents pass a URL
rather than image bytes or storage bucket/key fields.

Set `dry_run=true` to validate URL syntax and translation options and return a
structured plan without downloading, running OCR/translation, or uploading.
Dry-run does not resolve DNS, verify source accessibility, or check storage
permissions. Omitting it performs the real synchronous translation.
The MCP call requires an explicit translation direction. Auto-detection modes
use `any_<target>` with language or locale target codes:

| Mode | Target language | Font setting |
| --- | --- | --- |
| `any_zh_cn` | Simplified Chinese | `FONT_ZH_CN_FILE` |
| `any_zh_tw` | Traditional Chinese (Taiwan) | `FONT_ZH_TW_FILE` |
| `any_zh` | Simplified Chinese (compatibility alias) | `FONT_ZH_CN_FILE` |
| `any_en` | English | `FONT_LATIN_FILE` |
| `any_ja` | Japanese | `FONT_JA_FILE` |
| `any_ko` | Korean | `FONT_KO_FILE` |
| `any_fr` | French | `FONT_LATIN_FILE` |
| `any_ar` | Modern Standard Arabic | `FONT_ARABIC_FILE` |
| `any_de` | German | `FONT_LATIN_FILE` |
| `any_ru` | Russian | `FONT_CYRILLIC_FILE` |
| `any_nl` | Dutch | `FONT_LATIN_FILE` |
| `any_pt` | Portuguese | `FONT_LATIN_FILE` |
| `any_th` | Thai | `FONT_THAI_FILE` |
| `any_es` | Spanish | `FONT_LATIN_FILE` |
| `any_it` | Italian | `FONT_LATIN_FILE` |
| `any_vi` | Vietnamese | `FONT_VIETNAMESE_FILE` |
| `any_id` | Indonesian | `FONT_LATIN_FILE` |

`any_zh_cn` and `any_zh_tw` also process Chinese source text so they can convert
between Simplified and Traditional Chinese. The legacy `any_zh`, `en_zh`, and
`zh_en` directions remain supported, and the REST
endpoint retains its existing `en_zh` default. The configured `FONT_FILE` must
contain glyphs for the selected target language and remains the fallback when a
language-specific font variable is absent. `FONT_ZH_FILE` is retained as a
compatibility fallback for both Chinese variants when the newer variables are
unset. Font paths are read at startup, but
files are loaded only when that target language is rendered. Download or copy
the configured Noto files into the deployment before translating. Arabic uses
right-to-left shaping when Pillow was built with RAQM support.

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
