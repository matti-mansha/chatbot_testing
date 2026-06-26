# tools/ — standalone ops & dev utilities (not part of the running pipeline)

These are **manually-run CLIs** for debugging and maintenance. The live services do not
import or launch them, but they are genuinely useful during operation — keep them.

| File | Use |
|------|-----|
| `analyze_logs.py` | Parse/filter the structured service logs (`--stats`, `--errors`, `--search`, `--timeline`, `--api`). |
| `analyze_diagnostics.py` | Analyze Playwright diagnostic dumps (error/UI/network patterns, selector success). |
| `monitor_test_bot.py` | Health-check + stuck-session detector for the tester-bot API. |
| `check_notion_prompts.py` | Inspect the latest versioned prompts pulled from Notion. |
| `smoke_test_mila_open.py` | Smoke test for the Mila chat-open (FAB) state machine. |

Run from the repo root, e.g. `python tools/analyze_logs.py --stats`.
