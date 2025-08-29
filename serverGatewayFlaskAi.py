# ---------- imports ----------
# Flask primitives to handle HTTP
from flask import Flask, request, jsonify
# CORS middleware for browser calls
from flask_cors import CORS
# Read YAML config and prompts
import yaml
# Generate request ids for tracing
import uuid
# Measure request latency
import time
# Parse/validate JSON returned by AI
import json
# Optional regex fallback (config gated)
import re
# Type hints for clarity and safety
from typing import Any, Dict, Optional, Tuple
# SQLAlchemy engine + safe SQL   # explain: db libs
from sqlalchemy import create_engine, text
# type hint for engine           # explain: hint
from sqlalchemy.engine import Engine


# (OpenAI SDK; adjust if your environment uses a different client facade)
try:
    # Preferred 2024+ SDK client
    from openai import OpenAI
    # Flag for availability
    openaiClientAvailable = True
except Exception:
    # If not installed, we’ll raise clearly
    openaiClientAvailable = False


# ---------- configuration load ----------
with open("config.yaml", "r") as f:                           # Open top-level config YAML
    # Parse YAML into dict
    config = yaml.safe_load(f)

# read gateway key used in x-ligadata-api-key header  # explain: auth key
apiKeyExpected = config["security"]["apiKey"]


# get allowed origins  # explain: limit origins
allowedOrigins = config["server"].get(
    "allowedOrigins", ["https://chat.openai.com"])
# get port (default 8080)  # explain: server port
port = int(config["server"].get("port", 8080))

# ---------- db engine (PostgreSQL via SQLAlchemy) ----------
# read db config block          # explain: load yaml
dbCfg = config.get("database", {})
# optional fallback DSN string  # explain: fallback
databaseUrl = config.get("DATABASE_URL", "").strip()

# if full URL provided          # explain: use DSN
if databaseUrl:
    # use as-is                     # explain: dsn
    dsn = databaseUrl
    # timeout           # explain: timeout
    connectArgs = {"connect_timeout": int(dbCfg.get("connectTimeoutSec", 5))}
else:
    # host                          # explain: host
    host = dbCfg.get("host", "localhost")
    # port                          # explain: port
    port = int(dbCfg.get("port", 5432))
    # db name                       # explain: db
    name = dbCfg.get("name", "postgres")
    # user                          # explain: user
    user = dbCfg.get("user", "postgres")
    # password                      # explain: pass
    password = dbCfg.get("password", "")
    # compose DSN        # explain: compose
    dsn = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    # timeout            # explain: timeout
    connectArgs = {"connect_timeout": int(dbCfg.get("connectTimeoutSec", 5))}

# get ssl mode                  # explain: ssl
sslMode = dbCfg.get("sslMode", "disable")
# CA path if verify             # explain: CA
sslRootCertPath = dbCfg.get("sslRootCertPath", "")
# add ssl params if needed      # explain: ssl branch
if sslMode and sslMode != "disable":
    # set sslmode                   # explain: sslmode
    connectArgs["sslmode"] = sslMode
    # include CA path if set        # explain: CA path
    if sslRootCertPath:
        # pass CA path                  # explain: CA pass
        connectArgs["sslrootcert"] = sslRootCertPath

dbEngine: Engine = create_engine(                                    # create pooled engine          # explain: engine
    # DSN                           # explain: dsn
    dsn,
    # pool size                     # explain: pool
    pool_size=int(dbCfg.get("poolSize", 5)),
    # burst connections             # explain: burst
    max_overflow=int(dbCfg.get("poolOverflow", 2)),
    # keep connections fresh        # explain: ping
    pool_pre_ping=True,
    # timeouts/ssl                  # explain: args
    connect_args=connectArgs,
)

# ---------- app initialization ----------
# create Flask application instance           # explain: init app
app = Flask(__name__)

# define a simple health endpoint for Docker/Nginx


# add health check route                      # explain: health endpoint
@app.get("/healthz")
def healthz():                             # handler function                            # explain: handler
    # return JSON 200 so health checks pass       # explain: always healthy if app up
    return {"ok": True}, 200


# ---------- strict CORS ----------
CORS(                                       # enable CORS with explicit allow-list       # explain: enable CORS
    # target app                                 # explain: apply to app
    app,
    # only allow configured origins    # explain: restrict origins
    resources={r"/*": {"origins": allowedOrigins}},
    # no cookies/sessions needed                 # explain: no credentials
    supports_credentials=False,
    allow_headers=[                         # allow specific headers                     # explain: allowed headers
        "Content-Type", "Accept", "Origin",
        # your custom API key header                 # explain: custom auth header
        "x-ligadata-api-key"
    ],
    # limit methods to what you serve            # explain: allowed methods
    methods=["POST", "GET", "OPTIONS"],
    # cache preflight for 10 minutes             # explain: fewer preflights
    max_age=600
)

# (optional) a friendly root route for quick sanity checks


# add simple root                             # explain: root handler
@app.get("/")
def root():                                 # handler                                     # explain: handler
    # small JSON           # explain: small JSON
    return jsonify({"ok": True, "service": "mcp-gateway"}), 200


# AI provider/model settings block
aiConfig = config["ai"]
# Provider name (only 'openai' here)
aiProvider = aiConfig.get("provider", "openai")
aiModel = aiConfig.get("model", "gpt-4.1-mini")               # Model id to use
# API key for provider
aiApiKey = aiConfig.get("openaiApiKey", "")
aiTemperature = float(aiConfig.get("temperature", 0.0)
                      )       # Decoding temperature
aiTimeoutSeconds = int(aiConfig.get("timeoutSeconds", 12)
                       )    # Network timeout for AI call

enableRegexFallback = bool(config["routing"].get(
    "enableRegexFallback", True))  # Optional fallback

# ---------- ai router prompt load ----------
with open("ai/aiRouterPrompt.yaml", "r") as f:                # Open the router system prompt YAML
    # Parse into dict for structured use
    aiRouterPrompt = yaml.safe_load(f)

# ---------- optional precompiled regex fallback ----------
segCreateRegex = re.compile(                                  # Compile once; only used if enabled
    # Capture name with/without quotes
    r'\bcreate\s+segment\s+named\s+"?([A-Za-z0-9 _-]+)"?',
    # Case-insensitive for user friendliness
    flags=re.IGNORECASE
)                                                             # End regex

# ---------- utility: origin validation ----------


# Check if request Origin is allowed
def isOriginAllowed(origin: Optional[str]) -> bool:
    # If no Origin header (curl/server-to-server)
    if not origin:
        # Allow (no browser security needed)
        return True
    if "*" in allowedOrigins:                                 # If wildcard configured in YAML
        return True                                           # Allow any origin
    # Otherwise require exact match
    return origin in allowedOrigins

# ---------- domain service: segCreate ----------


# Backend business logic for segCreate
def segCreateService(segmentName: str) -> Dict[str, Any]:
    # Real world: persist to DB, call microservice, publish event, etc.
    # Human-friendly message for UI
    message = f'Segment "{segmentName}" is created'
    # Structured payload for the caller
    return {"segmentName": segmentName, "message": message}


# ---------- domain service: listSegments ----------
# return dict payload           # explain: func
def listSegmentsService() -> dict:
    # simple, indexed-friendly      # explain: sql
    sql = text("SELECT name FROM segments ORDER BY name ASC")
    # pooled connection             # explain: connect
    with dbEngine.connect() as conn:
        # fetch all rows                # explain: fetch
        rows = conn.execute(sql).fetchall()
    # first column -> name          # explain: parse
    names = [r[0] for r in rows]
    # structured result             # explain: result
    return {"segments": names, "count": len(names)}

# ---------- AI router: call model and parse structured JSON ----------


# Returns (intent, segmentName, confidence)
# Returns (intent, segmentName, confidence)
def callAiRouter(userPrompt: str) -> Tuple[str, Optional[str], float]:
    # Build strict instruction text from YAML
    systemText = (
        f"{aiRouterPrompt['task']}\n\n"
        f"{aiRouterPrompt['instructions']}\n\n"
        f"Supported intents: {', '.join(aiRouterPrompt['supportedIntents'])}\n\n"
        f"Schema example:\n{aiRouterPrompt['schemaExample']}\n\n"
        f"Edge cases:\n{aiRouterPrompt['edgeCases']}\n\n"
        f"Output constraints:\n{aiRouterPrompt['outputConstraints']}\n"
    )  # explain what JSON to return

    userText = f"USER_PROMPT:\n{userPrompt}"  # put raw user prompt

    if aiProvider != "openai":
        raise RuntimeError("Unsupported AI provider configured")  # safety

    if not openaiClientAvailable:
        raise RuntimeError(
            "openai SDK not installed; install `openai` package")  # safety

    # OpenAI client
    client = OpenAI(api_key=aiApiKey, timeout=aiTimeoutSeconds)

    # Use Chat Completions with JSON mode (more broadly available than Responses)
    cmpl = client.chat.completions.create(
        model=aiModel,  # e.g., gpt-4.1-mini or gpt-4o-mini
        temperature=aiTemperature,  # keep deterministic
        response_format={"type": "json_object"},  # force a JSON object
        messages=[
            # strict router instructions
            {"role": "system", "content": systemText},
            {"role": "user", "content": userText},      # raw user text
        ],
    )

    textOut = cmpl.choices[0].message.content  # the model’s JSON text

    try:
        obj = json.loads(textOut)  # parse JSON
    except Exception:
        return ("unknown", None, 0.0)  # degrade gracefully

    intent = str(obj.get("intent", "unknown")).strip()
    segmentName = obj.get("segmentName", None)
    if isinstance(segmentName, str):
        segmentName = segmentName.strip()
    confidence = float(obj.get("confidence", 0.0))

    return (intent, segmentName, confidence)

# ---------- tiny orchestrator: decide service using AI, with optional fallback ----------


# Return {"service":..., "args":...}
def chooseServiceAiFirst(userPrompt: str) -> Optional[Dict[str, Any]]:
    intent, segmentName, confidence = callAiRouter(
        userPrompt)  # Ask AI to classify + extract

    # Fast‑path: confident segCreate with a usable segmentName
    if intent == "segCreate" and segmentName:                 # If we have a clear segCreate intent
        # Route
        return {"service": "segCreate", "args": {"segmentName": segmentName}, "confidence": confidence}

    # list intent                   # explain: route
    if intent == "listSegments":
        return {"service": "listSegments", "args": {}, "confidence": confidence}

    # Optional fallback: regex as a safety net when AI is uncertain or failed
    if enableRegexFallback:                                   # Controlled by YAML flag
        # Try the deterministic pattern
        m = segCreateRegex.search(userPrompt)
        if m:                                                 # If matched
            # Extract and trim
            name = m.group(1).strip()
            if name:                                          # Ensure name not empty
                # Low but usable
                return {"service": "segCreate", "args": {"segmentName": name}, "confidence": 0.51}

    # No routing decision → caller will handle
    return None

# ---------- HTTP endpoint (called by GPT Action) ----------


# Define POST endpoint for GPT Action
@app.post("/mcp/route")
def routePrompt():                                            # Handler function
    t0 = time.time()                                          # Start latency timer

    # Read Origin header (browser calls)
    origin = request.headers.get("Origin", "")
    # Enforce CORS whitelist
    if not isOriginAllowed(origin):
        # Block disallowed origins
        return jsonify({"error": "CORS origin not allowed"}), 403

    apiKeyProvided = request.headers.get(
        "x-ligadata-api-key", "")  # Read API key header
    if apiKeyProvided != apiKeyExpected:                      # Check against expected key
        # Reject with 401 if invalid
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Parse JSON body (throw on invalid)
        body = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400        # Bad JSON → 400

    if not isinstance(body, dict):                            # Ensure JSON is an object
        # Enforce schema shape
        return jsonify({"error": "Body must be a JSON object"}), 400

    # Extract userPrompt
    userPrompt = (body.get("userPrompt") or "").strip()
    if not userPrompt:                                        # Require non-empty prompt
        # Missing → 400
        return jsonify({"error": "userPrompt is required"}), 400

    # Use provided or generate UUID
    requestId = (body.get("requestId") or str(uuid.uuid4()))

    try:
        # Ask AI router (with optional fallback)
        route = chooseServiceAiFirst(userPrompt)
    except Exception as e:
        # If router fails (e.g., SDK issue), fail gracefully with guidance
        responseObj = {
            # Not handled due to router failure
            "handled": False,
            # Show cause to operator
            "display": f"Routing failed: {str(e)}",
            # Include request id for tracing
            "data": {"requestId": requestId}
        }
        # Return graceful 200 with message
        return jsonify(responseObj), 200

    if not route:                                             # If nothing matched
        responseObj = {                                       # Build guidance response
            "handled": False,                                 # Not handled
            "display": "I couldn’t map that to a known action. Try: `create segment named \"VIP\"`.",  # Hint
            "data": {"requestId": requestId}                  # Trace id
        }
        # Return graceful success
        return jsonify(responseObj), 200

    # Extract chosen service name
    service = route["service"]
    # Extract arguments for the service
    args = route.get("args", {})
    # Surface router confidence
    confidence = route.get("confidence", 0.0)

    # Execute domain logic
    # existing branch               # explain: create
    if service == "segCreate":
        # call create                   # explain: call
        result = segCreateService(**args)
    # new list branch               # explain: list
    elif service == "listSegments":
        # run query                     # explain: call
        result = listSegmentsService()
    else:
        # fallback                      # explain: default#
        result = {"message": "serviceNotImplemented"}

    # End-to-end time in milliseconds
    elapsedMs = int((time.time() - t0) * 1000)

    displayText = (                                           # Markdown summary for GPT UI
        # Show service name
        f"**Service:** {service}\n"
        # Show router confidence
        f"**Confidence:** {confidence:.2f}\n"
        f"**Request ID:** {requestId}\n"                      # Correlation id
        f"**Latency:** {elapsedMs} ms\n\n"                    # Timing info
        # Human-friendly message
        f"**Result:** {result.get('message')}"
    )                                                         # End message composition

    responseObj = {                                           # Construct final response object
        "handled": True,                                      # We handled the request
        # Text for GPT UI to render
        "display": displayText,
        "data": {                                             # Machine-readable details
            "service": service,
            "confidence": confidence,
            "result": result
        }
    }                                                         # End response object

    # Return 200 OK with JSON
    return jsonify(responseObj), 200


# ---------- local dev entry-point ----------
if __name__ == "__main__":                                    # Allow running as script
    # Pull port from YAML
    port = int(config["server"]["port"])
    # For production, prefer gunicorn/uvicorn workers behind Nginx (TLS, gzip, timeouts)
    # Start Flask dev server (no debug)
    app.run(host="0.0.0.0", port=port, debug=False)
