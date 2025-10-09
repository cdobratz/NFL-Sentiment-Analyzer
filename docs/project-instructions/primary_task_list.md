Rules
- Complete one task at a time.
- Only change files needed for the specific task

Tasks to complete

[] 1 Configure file reviews.tools.github-checks in your project's settings in CodeRabbit to adjust the time to wait for GitHub Checks to complete.

[] 2 GitGuardian has uncovered 1 secret following the scan in your pull request.
The secret was in commit 5eb46fb in file tests/test_auth_endpoints.py Please consider investigating the findings and remediating the incidents. Failure to do so may lead to compromising the associated services or software components.

🔎 Detected hardcoded secret in your pull request
🛠 Guidelines to remediate hardcoded secrets

[] 3 .github/workflows/ci.yml around lines 184 to 204: the staging job's if condition
is checking for 'refs/heads/develop' which never matches this repo's branch
naming; update the condition to check for 'refs/heads/dev' so the staging job
runs on the intended branch, ensuring the rest of the step block remains
unchanged.

[] 4 In .github/workflows/ci.yml around lines 221 to 227, the Slack notify step uses
the unsupported input `webhook_url` for 8398a7/action-slack@v3 which causes the
step to fail; remove the `webhook_url` input and instead set the webhook via an
environment variable named SLACK_WEBHOOK_URL (e.g., add an `env:` block on that
step with SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}), or alternatively
swap to an action that accepts a webhook input; keep the other inputs (status,
channel) and the if: always() condition unchanged.

[] 5 In .kiro/specs/nfl-analyzer-improvements/requirements.md around line 44, fix the
capitalization and article in the user story: change "As a nfl fan" to "As an
NFL fan" so "NFL" is uppercase and the article is "an"; update that single line
accordingly to read "As an NFL fan, I want advanced NFL-specific sentiment
analysis so that I can make informed game decisions based on market sentim

[] 6 In app/api/admin.py around lines 1-12 and also apply fixes at lines 63-70,
91-96, and 122-129: the route handlers are using MongoDB "_id" fields with
user_id path parameters as plain strings which will cause queries/updates to
fail; import ObjectId and the validator (from bson.objectid import ObjectId,
InvalidId or use ObjectId.is_valid) at top, validate the incoming user_id and if
invalid raise HTTPException(status_code=400, detail="Invalid user id"), then
convert the string to an ObjectId before using it in any find_one/update/delete
query (e.g., collection.find_one({"_id": ObjectId(user_id)})); ensure all listed
handler locations perform the same validation/conversion and return 400 on
invalid ids.

[] 7 In app/api/api_keys.py around lines 7 and 20-24, the Pydantic model uses a
mutable default for the metadata field which can cause shared state across
requests; update the model to import Field from pydantic and replace the mutable
default dict with Field(default_factory=dict) for metadata (and any other
mutable defaults) so each instance gets its own fresh dict, keeping the rest of
the model unchanged.

[] 8 In app/api/api_keys.py around lines 79-83 (and similarly at lines 249-259),
current_user["_id"] is a Mongo ObjectId but the manager and API responses expect
a str; convert ObjectId to string when passing it into the manager (e.g.,
str(current_user["_id"])) and ensure any returned user id in responses is
serialized to a str before returning to the client. Replace direct usages of
current_user["_id"] with its string form in both the creation call and the
response payloads at the noted ranges.

[] 9 In app/api/data.py around lines 1009 to 1012, the loop is leaving the MongoDB
_id (an ObjectId) on each document which FastAPI can't serialize; after
assigning doc["id"] = str(doc["_id"]) remove the original _id (e.g.,
doc.pop("_id", None)) before appending to sentiment_data so only serializable
fields are returned. Ensure you do the pop on each doc in the async for loop (or
otherwise convert all ObjectId instances to strings) to prevent runtime 500s.

[] 10 In app/api/health.py around lines 85 to 104, db_manager.get_redis() may return
None which will cause an AttributeError when calling
asyncio.to_thread(redis_client.ping); before invoking any redis methods check if
redis_client is truthy and if not return a healthy-check failure payload like
{"status":"unhealthy","error":"Redis unavailable"} (or similar) so the exception
path is handled explicitly; if redis_client exists continue with the existing
to_thread calls and keep the existing exception logging for other failures.

[] 11 In app/api/sentiment.py around lines 77-90, doc_data is missing the fields
required by downstream aggregations; add "sentiment" (use result.sentiment),
"confidence" (use result.confidence), and "created_at" (use result.created_at if
available, otherwise datetime.utcnow()) to the dict so analytics functions that
filter/aggregate by created_at, sentiment, and confidence will see the inserted
documents.

[] 12 In app/api/sentiment.py around lines 152 to 174, the batch-inserted documents
lack a created_at field so they are excluded by consumers and get_data_summary;
add "created_at": datetime.utcnow() to each doc_data in the loop (mirroring the
single-document insertion path) and ensure the timestamp/metadata fields remain
consistent with the single-document flow (keep processed_at, model_version,
processing_time_ms, etc.).

[] 13 sentiment_service lacks the invoked accessors

sentiment_service.get_recent_sentiment and sentiment_service.get_team_sentiment are not defined on sentiment_service anywhere in the repo (confirmed by searching the service module). As soon as the WebSocket connection starts—or a client sends a subscribe message—this will raise AttributeError and drop the socket. Please either implement these methods on the service or call the existing API/helpers that return the same datasets before landing this.

🧰 Tools
🪛 Ruff (0.13.2)

[] 14 In app/core/config.py around lines 6 to 57, the nested Config class is legacy
Pydantic v1 and is ignored by pydantic v2 so .env values (e.g. mongodb_url,
secret_key) aren’t loaded; replace the nested Config with a v2-style model
config using SettingsConfigDict (import it) and assign it to model_config (e.g.
model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)) so
environment variables and .env are loaded correctly.

[] 15 In app/core/database_indexes.py around lines 285 to 293, the current except
Exception: block swallows all errors; change it to only catch database-related
exceptions by importing PyMongoError (from pymongo.errors import PyMongoError)
at the top of the module, replace except Exception as e: with except
PyMongoError as e: to record DB drop failures in results and the log, and allow
non-database exceptions (AttributeError, TypeError, etc.) to propagate so they
are not silently ignored.

[] 16 In app/core/database.py around lines 2-3 and 44-61, the async connect_redis
function (and related calls) currently imports and uses the synchronous redis
client which blocks the event loop; switch to the asyncio version by importing
from redis.asyncio (e.g., from redis.asyncio import Redis or
create_redis_client), instantiate the async Redis client rather than
redis.Redis, replace blocking calls like ping() with await client.ping(), and
use await client.close() (or await client.connection_pool.disconnect()) when
shutting down; update any type hints and error handling accordingly so all Redis
interactions are awaited and non-blocking.

[] 17 In app/core/dependencies.py around lines 13 to 55, the HTTPBearer security is
currently created without auto_error disabled which causes FastAPI to return a
403 before fallback flows run; change the security instantiation to
HTTPBearer(auto_error=False) and add an explicit check at the start of
get_current_user to reject missing credentials by raising the existing
credentials_exception when credentials is None or credentials.credentials is
falsy, then continue with the existing blacklist, decode and user lookup logic.

[] 18 In app/core/dependencies.py around lines 83 to 100, get_optional_user currently
calls get_current_user(credentials, db) without forwarding the Redis dependency,
which causes the Redis parameter inside get_current_user to be the Depends
sentinel and triggers "'Depends' object has no attribute 'get'"; fix by adding a
redis parameter to get_optional_user (e.g., redis=Depends(get_redis)) and pass
that redis through to get_current_user (call get_current_user(credentials, db,
redis)), ensuring the signature matches get_current_user's parameters.

[] 19 In app/core/middleware.py around lines 73 to 99, after decoding the Bearer JWT
and extracting user_id, populate request.state.user with the authenticated user
object (not just user_id) so downstream middleware like RateLimitMiddleware sees
the principal; to fix, load the user record (e.g., await
user_manager.get_by_id(user_id) or similar user lookup used elsewhere), assign
it to request.state.user and keep request.state.user_id as-is, and handle the
case where the lookup fails (leave request.state.user as None) to preserve
anonymous behavior.

[] 20 In app/core/monitoring.py around lines 235 to 245 the alert ID is built with
int(time.time()) which only has second resolution and can collide; replace the
timestamp component with a high-entropy identifier such as uuid.uuid4() (e.g.
f"{alert_type.value}_{uuid.uuid4()}") or use a microsecond timestamp to
guarantee uniqueness, and add the corresponding import (import uuid) at the top
of the file; ensure the final ID is converted to a string and maintain the same
Alert construction and append behavior.

[] 21 Unblock rate limiting with async Redis access.

db_manager.get_redis() returns the sync Redis client, so every await redis.get(...), await pipe.execute(), and await redis.ttl(...) raises TypeError. The exception handler then fails open, effectively disabling rate limiting. Call these sync methods via asyncio.to_thread (or switch to redis.asyncio) before merging.

[] 22 In app/core/rate_limiting.py around lines 164 to 167, the per-minute rate is
computed with rate_limit_per_hour // 60 which becomes 0 for hourly limits < 60;
update the expression to ensure at least one request per minute while preserving
the 100 cap by replacing the current min(rate_limit_per_hour // 60, 100) with
min(max(1, rate_limit_per_hour // 60), 100) so keys with small hourly limits
still allow one request per minute.

[] 23 In app/core/rate_limiting.py around lines 334 to 347, the code calls await
redis.ttl(redis_key) which fails because ttl is a synchronous/blocking call;
replace this with an async wrapper (e.g. ttl = await
asyncio.to_thread(redis.ttl, redis_key)) or switch to the async Redis client
method, ensure asyncio is imported, then compute reset_time exactly as before
using ttl if ttl > 0 else window_seconds and keep populating results unchanged.

[] 24 In app/main.py around lines 44 to 50, the broad except Exception handler
swallows asyncio.CancelledError which prevents shutdown; update the loop to
catch asyncio.CancelledError explicitly and re-raise it immediately before the
generic Exception handler so cancellation propagates; leave the existing
logger.error and sleep behavior for other exceptions unchanged.

[] 25 In app/models/nfl.py around lines 38-39 (and also at 59-60, 89, and 116), the
Pydantic model uses mutable default lists (e.g., aliases: List[str] = [] and
keywords: List[str] = []), which creates shared mutable state across instances;
change these to use default_factory to return a new list per instance (use
Field(default_factory=list) or the typing-compatible default_factory approach
for Pydantic), updating each affected field to use Field(default_factory=list)
instead of an empty list literal.

[] 26 In app/models/responses.py around line 5 and also covering lines 77-91 and
101-105, several Pydantic models use mutable default values (dict/list) which
can produce shared state across instances; replace any occurrences of default={}
or default=[] (or bare {} / []) with Field(default_factory=dict) or
Field(default_factory=list) as appropriate, add "from pydantic import Field" to
the imports at the top, and adjust the model fields to use those default_factory
calls so each instance gets its own fresh collection.

[] 27 In app/models/sentiment.py around lines 22 to 30, there is a duplicate
DataSource enum definition which conflicts with the one exported by
app/services/data_ingestion_service.py; remove this local enum and consolidate
to a single shared definition: choose one location (prefer central models
package e.g., app/models/sentiment.py), move/extend the enum there to include
all members used across the codebase (DRAFTKINGS, MGM, TWITTER, ESPN, NEWS,
BETTING, USER_INPUT, REDDIT, FANTASY, etc.), then update all modules to import
the shared DataSource from that single path (search-and-replace old imports),
update any serialization/validation type hints to reference the shared enum, and
run tests/lint to ensure no unresolved imports remain.

[] 28 In app/scripts/init_sentiment_db.py around lines 24-25, the code uses
settings.MONGODB_URL and settings.DATABASE_NAME which don't match the lowercase
attributes used elsewhere; change to settings.mongodb_url and
settings.database_name to align with DatabaseManager. In app/core/database.py
around lines ~44-60, connect_redis is using the synchronous redis client and
calling blocking methods inside an async function; switch to redis.asyncio
(import redis.asyncio as redis_asyncio), create the async client via
redis_asyncio.from_url, await ping(), and await close() in disconnect to avoid
blocking the event loop. In app/services/data_archiving_service.py around lines
~96-118 and ~174-197, insert_many is used without handling duplicate _id errors
which makes the job non‑idempotent (crash after insert but before delete will
cause duplicate-key BulkWriteError on retry); make archiving/deletion idempotent
by using ordered=False on inserts and catching BulkWriteError (ignore
duplicate-key 11000 errors or use upsert/replace semantics or transactions) and
count only actually-inserted documents; apply the same pattern for both archive
and deleted collections.

[] 29 In app/services/analytics_service.py around lines 47 to 56, the async
get_database() call may return None per app/core/database.get_database
docstring; before assigning to self.db and returning, check the awaited result
for None and handle it (do not cache None). If None, raise a clear exception
(e.g., RuntimeError or custom exception) with context like "failed to obtain
database connection" or perform the appropriate fallback, otherwise assign the
non-None AsyncIOMotorDatabase to self.db and return it.

[] 30 In app/services/analytics_service.py around lines 125 to 129, the code is using
the wrong collection name (db.sentiment_analysis) which is inconsistent with
migrations and index creation; change the reference to db.sentiment_analyses so
the service operates on the correct collection and indexes, and update any
related variable names or usages in the surrounding scope to match this
corrected collection name.

[] 31 In app/services/caching_service.py around lines 10-13 and additionally at 45-53,
118-123, 139-145, 162-166, 183-189, 202-205, 222-225 and 468-496, the code is
using synchronous redis operations against an async Redis client and treating
returned bytes as strings; this causes "coroutine was never awaited" and JSON
decode errors. Update imports/usages to use redis.asyncio client, await every
async Redis call (e.g., get, set, ping, info, scan_iter), and decode bytes
results (e.g., value.decode('utf-8')) before passing into json.loads; replace
any use of KEYS with an async SCAN/scan_iter pattern to avoid blocking, and
ensure all Redis interactions are awaited throughout the listed line ranges.

[] 32 In app/services/caching_service.py around lines 93 to 97, the code currently
uses pickle.loads to deserialize cached data which is unsafe; replace unpickling
with safe JSON deserialization: decode the hex string to bytes, convert bytes to
UTF-8 text, then use json.loads to parse into Python objects, handle
json.JSONDecodeError and UnicodeDecodeError by logging a clear error and
returning None, and ensure cached values are written as JSON (e.g., json.dumps)
elsewhere so reads/writes are consistent and safe.

[] 33 In app/services/data_archiving_service.py around lines 108-109 (and similarly at
186-187 and 348-350) the logger.info call is using len(batch) after the batch
has been cleared so it always logs 0; change the code to compute and store the
count before emptying/resetting the batch (e.g., count = len(batch)) and then
log that saved count, then proceed to clear/reset the batch so the logged number
reflects the actual archived batch size.

[] 34 In app/services/data_ingestion_service.py around lines 72 to 88, the rate-limit
logic uses timedelta.seconds (which ignores days) and naive local datetimes and
is not concurrency-safe; replace datetime.now() with timezone-aware UTC times
(e.g., datetime.now(timezone.utc) or datetime.utcnow() with awareness), use (now
- req_time).total_seconds() everywhere to compute age and wait_time so longer
windows work correctly, compute wait_time = max(0, self.time_window - (now -
oldest_request).total_seconds()), and serialize reads/writes to self.requests by
adding an asyncio.Lock on the instance and wrapping the trimming, capacity
check, sleep decision, and append inside a single async with self._requests_lock
block (release before awaiting sleep if you intentionally want other coroutines
to proceed, otherwise await inside the lock) so updates are atomic.

[] 35 In app/services/data_ingestion_service.py around lines 174-189, the code assumes
self.session exists and logs exceptions with logger.error; change it to lazily
create an aiohttp session if self.session is None before the request (e.g., call
or inline the same setup done in start()), then in the except block replace
logger.error with logger.exception so the traceback is recorded; ensure any
session created here is assigned to self.session so subsequent calls reuse it
and avoid leaking resources by keeping existing close semantics elsewhere.

[] 36 In app/services/data_ingestion_service.py around lines 561 to 579, the
collection_name is built with f"raw_{item.data_type}s" which pluralizes by
appending 's' and yields "raw_newss" for item.data_type == "news"; change this
to use an explicit mapping (e.g.,
{"tweet":"raw_tweets","news":"raw_news","game":"raw_games","betting_line":"raw_betting_lines"})
or a small helper that returns the correct collection name for each data_type,
then use that mapped collection_name for the duplicate lookup; keep the
unique_id logic but ensure the mapping covers all branches and return False if
data_type is unknown.

[] 37 Use the running loop when scheduling from the background thread

asyncio.get_event_loop() called inside the scheduler thread raises RuntimeError: There is no current event loop in thread ... on Python 3.11+, so every _schedule_* call fails and nothing ever reaches the queue. Capture the loop in start() (e.g., self.loop = asyncio.get_running_loop()) and reuse that reference inside each run_coroutine_threadsafe call. Repeat the loop reuse in the ESPNs, betting, and cleanup schedulers as well.

[] 38 In app/services/database_migration_service.py around lines 115-129, the code
treats failed migrations as applied because get_applied_migrations returns all
records; change the query to only return records where status == "applied"
(e.g., collection.find({"status": "applied"}) or equivalent filter) so applied =
await self.get_applied_migrations() yields only successfully applied migrations,
then compute applied_versions from that filtered list; also update any
current_version computation to derive the max/last version from this same
filtered applied set so failed migrations no longer affect pending detection or
current_version.

[] 39 In app/services/database_migration_service.py around lines 405 to 421, the
migration uses the wrong collection name db.sentiment_analysis; change all
references to db.sentiment_analyses so the update_many targets the existing
collection and avoids creating/updating an orphaned collection, i.e., replace
collection = db.sentiment_analysis with collection = db.sentiment_analyses and
update any other occurrences in this migration block accordingly.

[] 40 In app/services/mlops/huggingface_service.py around lines 5-11 (and also apply
same change to lines 88-114 and 109-111), the code calls heavy synchronous
Hugging Face APIs (pipeline(...) and from_pretrained(...)) directly and blocks
the event loop; wrap these blocking calls with asyncio.to_thread (or Starlette’s
run_in_threadpool) so they run in a threadpool and await their completion,
ensure any returned model/pipeline is awaited correctly, and keep the existing
imports (asyncio is already present) or add run_in_threadpool import if
preferred.

[] 41 
