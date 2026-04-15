# Integrating with loci-extract

The `loci-extract-api` HTTP service speaks plain multipart — any language,
any framework. This doc covers the wire format and ready-made client
snippets for common callers.

Default base URL on a LAN deployment: `http://<host>:8080` (adjust to
your host). All examples below use `http://10.10.100.51:8080` as a
placeholder.

---

## Endpoints

| Method | Path              | Purpose                                            |
|--------|-------------------|----------------------------------------------------|
| GET    | `/healthz`        | Liveness check, never requires auth                |
| GET    | `/capabilities`   | OCR engines available + LLM config + auth status   |
| GET    | `/docs`           | Swagger UI — interactive, ideal for manual tests   |
| GET    | `/openapi.json`   | OpenAPI schema — for client-code generation        |
| POST   | `/extract`        | One PDF → `Extraction` JSON / CSV / Lacerte / TXF  |
| POST   | `/extract/batch`  | Many PDFs → per-file results                       |

---

## Auth

By default: **open** (suitable for localhost and trusted LANs). Check
`/capabilities` — `auth_required: false` means no header required.

To lock it down, restart the server with `LOCI_EXTRACT_API_KEY` set:

```bash
pkill -f loci-extract-api
LOCI_EXTRACT_API_KEY=your-secret-key nohup loci-extract-api \
  --host 0.0.0.0 --port 8080 > /tmp/loci-extract-api.log 2>&1 &
disown
```

Clients then send `Authorization: Bearer your-secret-key` on every
request except `/healthz`.

---

## POST /extract — form fields

| Field          | Type             | Default     | Notes                                       |
|----------------|------------------|-------------|---------------------------------------------|
| `file`         | file (multipart) | required    | The PDF                                      |
| `format`       | string           | `json`      | `json` \| `csv` \| `lacerte` \| `txf`       |
| `model_url`    | string           | server env  | Override LLM endpoint per request            |
| `model_name`   | string           | server env  | Override model name per request              |
| `vision`       | bool             | `false`     | Send images to VLM instead of OCR            |
| `vision_model` | string           | server env  | VLM model name when `vision=true`            |
| `ocr_engine`   | string           | `auto`      | `tesseract` \| `easyocr` \| `paddleocr`     |
| `gpu`          | string           | `auto`      | `true` \| `false`                            |
| `dpi`          | int              | `300`       | OCR/render DPI                               |
| `redact`       | bool             | `true`      | Mask full SSN to last-4 on output            |
| `temperature`  | float            | `0.0`       | LLM sampling temperature                      |
| `max_tokens`   | int              | `4096`      | LLM response cap                             |
| `retry`        | int              | `2`         | Retries on invalid JSON                      |

## POST /extract/batch — form fields

Same as `/extract`, but `file` becomes `files` (repeat the field once
per PDF). Responds `{"results": [{"filename": "…", "documents": [...]}, ...]}`.

---

## Response shape (`format=json`)

```json
{
  "documents": [
    {
      "document_type": "W2",
      "tax_year": 2025,
      "data": {
        "employer": {"name": "...", "ein": "12-3456789", "address": "..."},
        "employee": {"name": "...", "ssn_last4": "XXX-XX-1234", "address": "..."},
        "federal": {"box1_wages": 75000.0, "box2_federal_withheld": 12000.0, "...": "..."},
        "box12": [{"code": "AA", "amount": 3833.36, "description": "..."}],
        "box13": {"retirement_plan": true},
        "box14_other": [{"label": "CA SDI", "amount": 92.16}],
        "state": [{"state_abbr": "CA", "box16_state_wages": 75000.0}],
        "local": []
      },
      "metadata": {
        "is_corrected": false,
        "is_summary_sheet": false,
        "notes": ["..."]
      }
    }
  ]
}
```

`document_type` is one of: `W2`, `1099-NEC`, `1099-MISC`, `1099-INT`,
`1099-DIV`, `1099-B`, `1099-R`, `1099-G`, `1099-SA`, `1099-K`, `1099-S`,
`1099-C`, `1099-A`, `1098`, `1098-T`, `1098-E`, `SSA-1099`, `RRB-1099`,
`K-1 1065`, `K-1 1120-S`, `K-1 1041`.

The `data` payload shape differs per `document_type`. Full per-type
schemas live in `loci_extract/schema.py` and `EXTRACT_SPEC.md`; the
runtime OpenAPI schema is at `/openapi.json`.

### CSV / Lacerte / TXF responses

- `Content-Type: text/csv` for `format=csv`
- `Content-Type: text/plain` for `format=lacerte` and `format=txf`

CSV has one row per document; nested fields (box12, state, transactions)
are serialized as JSON strings inside their cells. Lacerte and TXF are
v1-partial — W-2 + 1099-NEC/INT/DIV/R are supported; unsupported types
return HTTP 400 with a `detail` message so you can fall back to JSON.

---

## Client examples

### curl

```bash
# JSON
curl -F "file=@25-W2.pdf" http://10.10.100.51:8080/extract | jq

# CSV
curl -F "file=@25-W2.pdf" -F "format=csv" \
     http://10.10.100.51:8080/extract -o w2.csv

# Lacerte (tax-prep import)
curl -F "file=@25-W2.pdf" -F "format=lacerte" \
     http://10.10.100.51:8080/extract -o import.txt

# Vision mode for degraded scans
curl -F "file=@scan.pdf" -F "vision=true" -F "vision_model=qwen3-vl-32b" \
     http://10.10.100.51:8080/extract

# Batch
curl -F "files=@w2.pdf" -F "files=@1099.pdf" \
     http://10.10.100.51:8080/extract/batch

# With auth
curl -H "Authorization: Bearer your-secret-key" \
     -F "file=@25-W2.pdf" http://10.10.100.51:8080/extract
```

### Python (requests)

```python
import requests

r = requests.post(
    "http://10.10.100.51:8080/extract",
    files={"file": open("25-W2.pdf", "rb")},
    data={"format": "json"},
    headers={"Authorization": "Bearer your-secret-key"} if AUTH else {},
    timeout=300,  # LLM calls can take 30-90s per page
)
r.raise_for_status()
extraction = r.json()
for doc in extraction["documents"]:
    print(doc["document_type"], doc["tax_year"])
```

### Python (httpx async)

```python
import httpx

async def extract(path: str) -> dict:
    async with httpx.AsyncClient(timeout=300) as client:
        with open(path, "rb") as f:
            r = await client.post(
                "http://10.10.100.51:8080/extract",
                files={"file": f},
            )
        r.raise_for_status()
        return r.json()
```

### Node / TypeScript

```ts
const form = new FormData();
form.append("file", fileBlob, "25-W2.pdf");
form.append("format", "json");

const res = await fetch("http://10.10.100.51:8080/extract", {
  method: "POST",
  body: form,
  headers: AUTH ? { Authorization: `Bearer ${apiKey}` } : {},
});
if (!res.ok) throw new Error(await res.text());
const extraction = await res.json();
```

### Go

```go
body := &bytes.Buffer{}
w := multipart.NewWriter(body)
fw, _ := w.CreateFormFile("file", "25-W2.pdf")
f, _ := os.Open("25-W2.pdf"); defer f.Close()
io.Copy(fw, f)
w.WriteField("format", "json")
w.Close()

req, _ := http.NewRequest("POST", "http://10.10.100.51:8080/extract", body)
req.Header.Set("Content-Type", w.FormDataContentType())
resp, _ := http.DefaultClient.Do(req)
defer resp.Body.Close()
```

### PowerShell

```powershell
$form = @{
  file   = Get-Item "25-W2.pdf"
  format = "json"
}
Invoke-RestMethod -Uri http://10.10.100.51:8080/extract -Method Post -Form $form
```

### Make / n8n / Zapier

Use an **HTTP Request** node:

- Method: `POST`
- URL: `http://10.10.100.51:8080/extract`
- Body type: `multipart/form-data`
- Fields:
  - `file` (binary — bind to the file object from the previous node)
  - `format` (text: `json`)
- (optional) Header: `Authorization: Bearer <key>`

---

## OpenAPI client code generation

The server publishes the schema at `/openapi.json`. Use any OpenAPI
generator for fully typed clients:

```bash
# TypeScript types
npx openapi-typescript http://10.10.100.51:8080/openapi.json -o loci-extract.d.ts

# Python client
openapi-python-client generate --url http://10.10.100.51:8080/openapi.json

# Go client
oapi-codegen -package lociextract \
  http://10.10.100.51:8080/openapi.json > loci-extract.go
```

---

## Typical integration patterns

1. **Batch tax-prep workflow.** Watch a drop folder; POST each PDF as it
   arrives; write the resulting `.txt` (Lacerte) or `.txf` to an import
   queue your tax software picks up.
2. **Document management plugin.** On document upload, DMS hook calls
   `/extract`, stores the returned JSON alongside the PDF.
3. **Chatbot.** User DMs a W-2 attachment; bot forwards to `/extract`,
   replies with a formatted summary.
4. **Agent tool.** Expose `/extract` as a tool an LLM agent can call
   with any tax PDF it encounters and get back structured JSON.

---

## Practical notes

- **Latency.** Each PDF is one LLM call. With Qwen3-VL 32B on llama.cpp:
  ~5–15s for a clean single-page W-2; 30–90s per page in `vision=true`
  mode; 60s+ for multi-page 1099-B or K-1. **Set client timeouts to at
  least 300s.**
- **Parallelism.** The server processes requests synchronously — to
  increase throughput, run multiple concurrent HTTP requests; the
  LLM backend handles queuing.
- **Max upload.** 50 MB per file by default. Override with
  `LOCI_EXTRACT_MAX_UPLOAD_BYTES` env var on the server.
- **PDFs only.** The API currently expects PDF input. Image-only files
  (JPG, PNG, TIFF) need to be converted first (`img2pdf scan.jpg -o scan.pdf`).
- **Error codes.**
  - `400` — bad format or unsupported doc type for the requested format
  - `401` — missing/wrong Bearer token (when auth enabled)
  - `413` — upload too large
  - `500` — LLM failure (retries exhausted); the `detail` field carries
    the underlying exception message

---

## Quick reachability check from a peer machine

```bash
curl http://10.10.100.51:8080/healthz
# expect: {"status":"ok","service":"loci-extract","version":"0.2.0"}
```

If that 404s or connection-refuses:

```bash
ss -tlnp | grep 8080           # server bound to 0.0.0.0, not 127.0.0.1?
sudo ufw status                 # firewall allowing the port?
```

---

## See also

- `EXTRACT_SPEC.md` — full schema reference for every document type
- `README.md` — install + CLI usage
- `/docs` — live Swagger UI served by the API itself
