# Rights and source eligibility

Discovering a video never grants permission to ingest or transform it. Every project carries
rights decisions (`rights_decisions` table, newest wins) with: category, status, licence,
owner, permission evidence, attribution requirement and text, allowed transformations,
commercial-use status, who decided, and when.

| Category | Gate result |
|---|---|
| USER_OWNED, USER_UPLOADED, PUBLIC_DOMAIN | approved |
| CREATIVE_COMMONS | approved; needs a human if marked non-commercial |
| CREATOR_PROVIDED, EXPLICITLY_LICENSED | approved only with recorded permission evidence, else needs a human |
| UNKNOWN | needs a human (`rights.block_unknown: true`) |
| REFERENCE_ONLY | rejected for ingestion (may be studied, never rendered) |
| REJECTED | rejected |

"Needs a human" leaves the project in `RIGHTS_PENDING` with a `rights_ambiguity` approval
request. Decide with `POST /projects/{id}/rights` (`approve`, optionally a corrected
category/licence/evidence) or `POST /approvals/{id}`; the decision is audited.

Studio never downloads videos from YouTube or any platform. A `youtube` source without a
supplied file fails ingestion with an explanation. Supply the permitted file by upload
(`POST /projects/{id}/source`) or a path the worker can read.

Attribution text from the rights decision is added to the drafted video description.
