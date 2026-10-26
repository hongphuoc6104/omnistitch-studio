# Immediate terminal Drive fetch

`python3 tools/colab_fetch.py --config /absolute/path/config.json` runs one
interactive terminal workflow. It checks profile identity, allocates a uniquely
named **CPU** session, executes a real kernel probe, then invokes the usual
`colab-profile PROFILE drivemount --session NAME`. The user completes Google
consent in the correct browser account and keeps the terminal open. No token,
OTP or OAuth code is supplied to this helper or recorded in its status file.

After mount returns, the helper immediately starts an uploaded Python file as a
separate child process. This avoids notebook-cell `__file__` assumptions and a
long-lived execution WebSocket. File API polling reads incremental status;
collection has a bounded deadline. Transfers use 16 MiB parts, 4 MiB streaming
buffers, part hashes, total size and the **existing source manifest's** SHA256.
The downloaded archive is not extracted or executed.

Example configuration (use actual known values and paths):

```json
{
  "profile": "my-profile",
  "email": "researcher@example.org",
  "source": "/content/drive/MyDrive/project/proxy_pack.tar.gz",
  "manifest": "/workspace/audit/proxy_pack_manifest.json",
  "destination": "/workspace/audit/verified_proxy_pack.tar.gz",
  "session_prefix": "project-drive-fetch"
}
```

The source manifest contains `archive_sha256`, positive `archive_bytes` and
`parts` whose byte counts sum to the archive size. Do not retype a checksum into
code. Archive identity comes from independently retained source evidence, not a
hash generated only after download.

The temporary directory shares the destination filesystem. Exclusive hard-link
publication refuses to replace a concurrently created output. An existing
correct file is reused without allocating a runtime; an existing different file
requires a new destination. This path targets filesystems supporting hard links.

The helper stops only the uniquely named runtime it created, including mount,
worker, timeout, checksum and cancellation failures. It never stops unrelated
sessions. If cleanup fails, the verified archive remains usable, status records
the cleanup problem and the command does not claim complete lifecycle success.
`DESTINATION.fetch.json` records the account, owned session, source, checksum and
verification/cleanup state, with no proxy tokens. CLI error query strings are
redacted; the interactive mount authorization URL remains visible only in the
user's terminal as part of the normal consent flow.

This workflow avoids handing the user a runtime name that can become stale
between chat turns. It does not guarantee unlimited VM lifetime or prevent
backend allocation, authentication, networking or capacity failures. A kernel
probe passing does not guarantee that later Drive operations will succeed.

The file-child E2E suite exercises actual temporary files and a Python child
through a simulated CLI transport. It covers successful SHA verification and
cleanup, corrupt source/download rejection, wrong-account preflight, mount
failure, existing-file reuse, containment/manifest guards, concurrent publication
and cleanup failure. It does **not** claim a real Google authorization or Drive
transfer has been completed; that final flow requires the user's terminal.
