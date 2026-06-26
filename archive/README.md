# archive/ — dead / superseded code (not part of the running system)

These files are **not used by the live pipeline** and are kept only for reference/history.
Nothing in the running pipeline imports or launches them.

| File | Why it's here |
|------|---------------|
| `playwright_bridge_bot.py` | Legacy **non-headless** bridge. Superseded by `playwright_bridge_bot_headless.py`. |
| `dashboard_app.py` | Streamlit KPI dashboard. **Redundant** — the KPI requirement is now met by emitting KPI JSON in the conversation/prompt for the analytics vendor, not by a UI. |
| `calculate_kpis.py` | KPI aggregation. Imported **only** by `dashboard_app.py`, so it retires with the dashboard. |
| `test_chatbot_app.py` | Early Streamlit prototype that predates the prepare→execute→evaluate pipeline. |

Safe to delete entirely once the handover is signed off; archived for now so nothing is lost.
