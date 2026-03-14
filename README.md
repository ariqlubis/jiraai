## JiraAI

CLI that uses Vertex AI (Gemini) to create and update Jira tickets from natural language.

### What it does

- `engineer` mode: guided flow that drafts a ticket, then asks you to confirm before writing to Jira
- `pm` mode: interactive chat to create/update/list tickets (create/update draft first, then confirm)
- Optional: create subtasks under a parent ticket when you ask for a breakdown

### Requirements

- Python 3.11+
- Jira credentials
  - Jira Cloud: `JIRA_USERNAME` is your email, `JIRA_PASSWORD` is an API token
- Google Cloud project with Vertex AI access

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (start from `.env.example`) and fill in values.

If you want the notebook dependencies too:

```bash
pip install -r requirements-notebooks.txt
```

### Usage

Engineer mode (creates a ticket):

```bash
python main.py engineer
```

PM chat mode:

```bash
python main.py pm
```

Draft + confirm (writes):

- Create/update operations will always show a draft first.
- Reply `yes`/`confirm` to apply the draft to Jira.
- Reply `no`/`cancel` to discard the draft.

Autocomplete:

- In `pm` and `engineer` modes, type `@` then press `Tab` to autocomplete file paths.
- Autocomplete resolves relative paths from the project root (same folder as `main.py`).
The modern UI also shows a boxed input area while typing.

Subtasks:

- Ask explicitly: “Create a ticket and add 3 subtasks” or “Add subtasks to PER-4”.
- Subtasks are always drafted first and only created after confirmation.

PRD / PDF attachments:

- Reference a PDF anywhere in your prompt using `@/path/to/file.pdf`.
- Limit pages with `#pages=`: `@/path/to/prd.pdf#pages=1-3,7`.
- The agent extracts text and uses it to draft the description.

Tone:

```bash
python main.py pm --tone pro
python main.py pm --tone snarky
```

### Configuration

Env vars:

- `JIRA_URL` (example: `https://your-domain.atlassian.net`)
- `JIRA_USERNAME`
- `JIRA_PASSWORD`
- `PROJECT_ID` (Google Cloud project id)
- `LOCATION` (default: `us-central1`)
- `VERTEX_MODEL` (default: `gemini-3.1-flash-lite-preview`)
- `JIRA_USE_ADF` (optional: `true/false`, overrides auto-detect for Jira Cloud vs Server)
- `JIRAAI_TONE` (optional: `pro` or `snarky`)
- `JIRAAI_ATTACHMENT_MAX_CHARS` (optional: max characters pulled from PDFs; default 8000)
- `JIRAAI_UI` (optional: `modern` or `classic`; default `modern`)

### Troubleshooting

- `PROJECT_ID environment variable is not set`: ensure `.env` exists and has `PROJECT_ID`.
- Jira Cloud description errors: set `JIRA_USE_ADF=true` (or leave it unset and use a `.atlassian.net` URL).
- Jira user assignment may fail if your instance hides emails; use a Jira-visible email and confirm permissions.
