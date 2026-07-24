# Runtime and recognition logs

The live pipeline stores two complementary log types.

## Runtime log

Console output is also written to:

```text
logs/cctv_ai.log
```

The file rotates at 5 MB and retains five backups.

View the latest lines in PowerShell:

```powershell
Get-Content logs\cctv_ai.log -Tail 50
```

Follow the file live:

```powershell
Get-Content logs\cctv_ai.log -Wait
```

The live pipeline periodically writes a `FACE_DIAGNOSTICS` aggregate with
counts for identity-ready faces, accepted matches, conflicts rejected on an
already-confirmed track, below-threshold matches, ambiguous matches, faces
rejected as too small, missing faces and detector/embedding failures.
Per-track rejection detail is available at `LOG_LEVEL=DEBUG`; routine runs use
the on-screen explanation to avoid writing one log record per frame.

## Structured SQLite events

The `recognition_events` table stores only:

- `TRACK_STARTED`
- `IDENTITY_CONFIRMED`
- `TRACK_ENDED`

Every application run receives a unique `session_id`, because ByteTrack IDs
can restart from the same values in a later run.

Initialize or migrate the schema:

```powershell
python scripts\init_db.py
```

Run recognition normally:

```powershell
python scripts\live_recognition.py --source 0 --show-faces
```

View recent events:

```powershell
python scripts\view_logs.py --limit 50
```

View only confirmations:

```powershell
python scripts\view_logs.py --event-type IDENTITY_CONFIRMED
```

Export recent events:

```powershell
python scripts\view_logs.py --limit 1000 --csv outputs\recognition_events.csv
```

Disable structured SQLite logging for a particular run:

```powershell
python scripts\live_recognition.py --source 0 --disable-event-logs
```
