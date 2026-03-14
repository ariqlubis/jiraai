import vertexai
import os
from typing import Optional
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Tool,
    FunctionDeclaration,
    Part,
    Content
)

from dotenv import load_dotenv
from loguru import logger
from agent.tools import (
    create_jira,
    create_subtasks,
    get_ticket,
    update_ticket_fields,
    list_tickets
)
from agent.attachments import expand_attachments

import warnings
warnings.filterwarnings('ignore')

load_dotenv()

_VERTEX_INITIALIZED = False


def _init_vertex() -> None:
    global _VERTEX_INITIALIZED
    if _VERTEX_INITIALIZED:
        return

    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "PROJECT_ID environment variable is not set. "
            "Create a .env file (see .env.example) and set PROJECT_ID / LOCATION."
        )

    vertexai.init(
        project=project_id,
        location=os.getenv("LOCATION", "us-central1"),
    )
    _VERTEX_INITIALIZED = True


_TOOL = Tool(
    function_declarations=[
        FunctionDeclaration(
            name='create_ticket',
            description=(
                "Create a new Jira ticket. Infer issue_type from context: "
                "'Bug' for errors/crashes/broken behaviour"
                "'Story' for user-facing features, "
                "'Task' for chores/infra/refactors"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "project_key": {"type": "string", "description": "Jira project key, e.g. `PROJ`"},
                    "title": {"type": "string", "description": "Short, clear ticket summary"},
                    "description": {"type": "string", "description": "Detailed description in plain text"},
                    "issue_type": {"type": "string", "enum": ["Bug", "Story", "Task"]},
                    "priority": {"type": "string", "enum": ["Highest", "High", "Medium", "Low", "Lowest"]},
                    "assignee_email": {"type": "string", "description": "Optional assignee email"},
                    "subtasks": {
                        "type": "array",
                        "description": "Optional list of subtasks to create under the new ticket.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "priority": {"type": "string", "enum": ["Highest", "High", "Medium", "Low", "Lowest"]},
                                "assignee_email": {"type": "string"},
                            },
                            "required": ["title", "description"],
                        },
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Set true only after the user explicitly confirms the draft.",
                    },
                },
                "required": ["project_key", "title", "description", "issue_type", "priority"]
            }
        ),
        FunctionDeclaration(
            name="get_ticket",
            description="Fetch details of an existing Jira ticket by its key",
            parameters={
                "type": "object",
                "properties": {
                    "ticket_key": {"type": "string", "description": "Jira ticket key, e.g. 'PROJ-32'"}
                },
                "required": ["ticket_key"]
            }
        ),
        FunctionDeclaration(
            name="update_ticket",
            description="Update fields on an existing Jira ticket",
            parameters={
                "type": "object",
                "properties": {
                    "ticket_key": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["Highest", "High", "Medium", "Low", "Lowest"]},
                    "assignee_email": {"type": "string"},
                    "confirm": {
                        "type": "boolean",
                        "description": "Set true only after the user explicitly confirms the draft.",
                    },
                },
                "required": ['ticket_key']
            }
        ),
        FunctionDeclaration(
            name="list_tickets",
            description="List jira tickets in a project, optionally filtered by status or assignee",
            parameters={
                "type": "object",
                "properties": {
                    "project_key": {"type": "string"},
                    "status": {"type": "string", "description": "e.g. 'To Do', 'In Progress', 'Done'"},
                    "assignee_email": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Max tickets to return (default 10)"}
                },
                "required": ["project_key"]
            }
        ),
        FunctionDeclaration(
            name="create_subtasks",
            description="Create one or more subtasks under an existing Jira ticket",
            parameters={
                "type": "object",
                "properties": {
                    "parent_key": {"type": "string", "description": "Parent Jira ticket key, e.g. 'PROJ-32'"},
                    "subtasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "priority": {"type": "string", "enum": ["Highest", "High", "Medium", "Low", "Lowest"]},
                                "assignee_email": {"type": "string"},
                            },
                            "required": ["title", "description"],
                        },
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Set true only after the user explicitly confirms the draft.",
                    },
                },
                "required": ["parent_key", "subtasks"],
            },
        ),
    ]
)


def _is_true(value: object) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _dispatch(name: str, args: dict, allow_writes: bool) -> dict:
    try:
        if name == 'create_ticket':
            confirmed = _is_true(args.get("confirm"))
            if not confirmed or not allow_writes:
                proposed_args = {
                    "project_key": args.get("project_key"),
                    "title": args.get("title"),
                    "description": args.get("description"),
                    "issue_type": args.get("issue_type"),
                    "priority": args.get("priority"),
                    "assignee_email": args.get("assignee_email"),
                    "subtasks": args.get("subtasks"),
                }
                result = {
                    "needs_confirmation": True,
                    "action": "create_ticket",
                    "proposed_args": proposed_args,
                    "message": "Draft only. Ask user to confirm before creating the Jira ticket.",
                }
                return result

            result = create_jira(
                project_key=args["project_key"],
                title=args['title'],
                description=args['description'],
                issue_type=args['issue_type'],
                priority=args.get("priority", "Medium"),
                assignee_email=args.get("assignee_email"),
                subtasks=args.get("subtasks"),
            )
        elif name == 'get_ticket':
            result = get_ticket(
                ticket_key=args['ticket_key']
            )
        elif name == 'update_ticket':
            confirmed = _is_true(args.get("confirm"))
            if not confirmed or not allow_writes:
                proposed_args = {
                    "ticket_key": args.get("ticket_key"),
                    "title": args.get("title"),
                    "description": args.get("description"),
                    "priority": args.get("priority"),
                    "assignee_email": args.get("assignee_email"),
                }
                result = {
                    "needs_confirmation": True,
                    "action": "update_ticket",
                    "proposed_args": proposed_args,
                    "message": "Draft only. Ask user to confirm before updating the Jira ticket.",
                }
                return result

            result = update_ticket_fields(
                ticket_key=args['ticket_key'],
                title=args.get('title'),
                description=args.get('description'),
                priority=args.get("priority"),
                assignee_email=args.get('assignee_email')
            )
        elif name == 'list_tickets':
            result = list_tickets(
                project_key=args['project_key'],
                status=args.get("status"),
                assignee_email=args.get("assignee_email"),
                max_results=args.get("max_results", 10)
            )
        elif name == "create_subtasks":
            confirmed = _is_true(args.get("confirm"))
            if not confirmed or not allow_writes:
                proposed_args = {
                    "parent_key": args.get("parent_key"),
                    "subtasks": args.get("subtasks"),
                }
                result = {
                    "needs_confirmation": True,
                    "action": "create_subtasks",
                    "proposed_args": proposed_args,
                    "message": "Draft only. Ask user to confirm before creating subtasks.",
                }
                return result

            result = create_subtasks(
                parent_key=args.get("parent_key"),
                subtasks=args.get("subtasks") or [],
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        result = {"error": str(e)}

    return result


SYSTEM_PROMPT_SNARKY = """
You're Jira AI, a snarky but competent AI assistant that helps teams manage Jira tickets.

Your personality:
- You roast vague or lazy requests (briefly, then get to work)
- You are direct and efficient - no filler
- You ALWAYS complete the task after any roast.

Your capabilities:
- Create, read, update, and list Jira tickets.
- Create subtasks under a parent ticket when asked.
- Infer ticket type, priority, and description from natural language
- Ask clarifying questions only when project_key is truly missing

Rules:
- Never hallucinate ticket keys or project keys
- When creating tickets, always infer a sensible priority if not given
- Write description that are actually useful to a developer / engineer
- If the user says "assign to me/myself", ask for their assignee email.

"""

SYSTEM_PROMPT_PRO = """
You're Jira AI, a concise and competent assistant that helps teams manage Jira tickets.

Your personality:
- Professional and direct
- No roasting, no sarcasm, no filler

Your capabilities:
- Create, read, update, and list Jira tickets.
- Create subtasks under a parent ticket when asked.
- Infer ticket type, priority, and description from natural language.
- Ask clarifying questions only when project_key is truly missing.

Rules:
- Never hallucinate ticket keys or project keys.
- When creating tickets, always infer a sensible priority if not given.
- Write descriptions that are actually useful to a developer / engineer.
- If the user says "assign to me/myself", ask for their assignee email.
"""

CONFIRMATION_RULES = """
Write safety (must follow):
- Never write to Jira immediately for create/update.
- First draft the fields by calling create_ticket/update_ticket with confirm=false.
- Present the draft clearly and ask for explicit confirmation (user replies "yes"/"confirm").
- Only after explicit confirmation, call the same tool again with confirm=true.
- If the user replies "no"/"cancel", do not write.
""".strip()

ATTACHMENT_RULES = """
If the user includes attachments (look for the "Attachments:" section), use their content to craft the title and description.
Summarize or extract only the relevant parts; do not quote excessively.
""".strip()

MODE_GUIDE_ENGINEER = """
Engineer mode description template (implementation-focused):
Summary
Context
Proposed Approach
Tech Stack
Acceptance Criteria
Test Plan
Observability
Risks / Rollback
Links / References

If issue_type is Bug, prioritize:
Repro Steps, Expected vs Actual, Environment, Logs/Artifacts, Impact.

If the user asks for a breakdown, propose a small set of subtasks with clear titles.
""".strip()

MODE_GUIDE_PM = """
PM mode description template (spec-lite):
Problem
Goals
Non-Goals
Scope
User Stories
Acceptance Criteria
Dependencies
Rollout / Comms
Metrics
Open Questions

If issue_type is Bug, prioritize:
User impact, repro, severity, and a clear definition of done.

If the user asks for a breakdown, propose a small set of subtasks with clear titles.
""".strip()


def _build_system_prompt(mode: str, tone: str) -> str:
    base = SYSTEM_PROMPT_SNARKY if (tone or "").lower() == "snarky" else SYSTEM_PROMPT_PRO
    mode_key = (mode or "pm").strip().lower()
    mode_guide = MODE_GUIDE_ENGINEER if mode_key == "engineer" else MODE_GUIDE_PM
    return "\n\n".join([base.strip(), CONFIRMATION_RULES, ATTACHMENT_RULES, mode_guide]).strip()


def _format_draft_preview(tool_result: dict) -> str:
    action = (tool_result.get("action") or "").strip()
    args = tool_result.get("proposed_args") or {}

    lines = ["### Draft (not applied to Jira yet)"]

    if action == "create_ticket":
        project_key = args.get("project_key") or ""
        if project_key:
            lines.append(f"**Project**: `{project_key}`")
    elif action == "update_ticket":
        ticket_key = args.get("ticket_key") or ""
        if ticket_key:
            lines.append(f"**Ticket**: `{ticket_key}`")
    elif action == "create_subtasks":
        parent_key = args.get("parent_key") or ""
        if parent_key:
            lines.append(f"**Parent Ticket**: `{parent_key}`")

    title = args.get("title")
    if title:
        lines.append(f"**Title**: {title}")

    issue_type = args.get("issue_type")
    if issue_type:
        lines.append(f"**Type**: `{issue_type}`")

    priority = args.get("priority")
    if priority:
        lines.append(f"**Priority**: `{priority}`")

    assignee_email = args.get("assignee_email")
    if assignee_email:
        lines.append(f"**Assignee**: `{assignee_email}`")

    description = args.get("description")
    if description:
        lines.append("")
        lines.append("**Description preview**:")
        lines.append("```text")
        lines.append(str(description).rstrip())
        lines.append("```")

    subtasks = args.get("subtasks") or []
    if subtasks:
        lines.append("")
        lines.append("**Subtasks**:")
        for idx, subtask in enumerate(subtasks, start=1):
            title = (subtask or {}).get("title") or ""
            sub_desc = (subtask or {}).get("description") or ""
            assignee = (subtask or {}).get("assignee_email")
            line = f"{idx}. {title}".strip()
            if assignee:
                line += f" (assignee: {assignee})"
            lines.append(line)
            if sub_desc:
                lines.append(f"   - {sub_desc}")

    lines.append("")
    lines.append("**Confirm?** Reply `yes` to apply, `no` to cancel, or tell me what to change.")
    return "\n".join(lines)


def _format_draft_previews(tool_results: list[dict]) -> str:
    if not tool_results:
        return "No draft available."
    if len(tool_results) == 1:
        return _format_draft_preview(tool_results[0])

    blocks = []
    for idx, result in enumerate(tool_results, start=1):
        blocks.append(f"## Draft {idx}")
        blocks.append(_format_draft_preview(result))
    blocks.append("")
    blocks.append("**Confirm?** Reply `yes` to apply all drafts, `no` to cancel, or tell me what to change.")
    return "\n".join(blocks)


def run_agent(
    user_message: str,
    history: Optional[list] = None,
    tone: str = "pro",
    mode: str = "pm",
    allow_writes: bool = False,
    return_meta: bool = False,
) -> tuple:
    """
    Run one turn of the agentic loop.

    Args:
        user_message: The user's latest message
        history: List of previous Content objects (for multi-turn chat)

    Returns:
        (assistant_reply_text, updated_history)
    """
    _init_vertex()
    system_prompt = _build_system_prompt(mode=mode, tone=tone)
    model = GenerativeModel(
        model_name=os.getenv("VERTEX_MODEL", "gemini-3-flash-preview"),
        system_instruction=system_prompt,
        tools=[_TOOL],
        generation_config=GenerationConfig(
            temperature=0.1,
            candidate_count=1,
        ),
    )

    history = history or []
    expanded_message, _attachment_errors = expand_attachments(user_message)
    history.append(Content(role='user', parts=[Part.from_text(expanded_message)]))

    meta = {
        "needs_confirmation": False,
        "write_performed": False,
        "last_tool": None,
    }

    while True:
        response = model.generate_content(history)
        candidate = response.candidates[0]
        history.append(candidate.content)

        fn_calls = []
        for part in candidate.content.parts:
            fn_call = getattr(part, "function_call", None)
            if fn_call and fn_call.name:
                fn_calls.append(fn_call)

        if not fn_calls:
            text_parts = [
                part.text
                for part in candidate.content.parts
                if hasattr(part, 'text') and part.text
            ]
            reply = "\n".join(text_parts)
            if return_meta:
                return reply, history, meta
            return reply, history
        
        tool_response_parts = []
        draft_results = []
        for fn_call in fn_calls:
            fn_name = fn_call.name
            fn_args = dict(fn_call.args)
            logger.info(f"Calling tool: {fn_name}({fn_args})")
            tool_result = _dispatch(fn_name, fn_args, allow_writes=allow_writes)
            logger.debug(f"Tool result: {tool_result}")

            tool_response_parts.append(
                Part.from_function_response(
                    name=fn_name,
                    response=tool_result,
                )
            )

            meta["last_tool"] = fn_name
            if tool_result.get("needs_confirmation") is True:
                draft_results.append(tool_result)

            if (
                fn_name in {"create_ticket", "update_ticket", "create_subtasks"}
                and not tool_result.get("error")
                and not tool_result.get("needs_confirmation")
            ):
                # A successful write will only happen when allow_writes=True and confirm=true.
                meta["write_performed"] = True

        history.append(Content(role="user", parts=tool_response_parts))

        if draft_results:
            meta["needs_confirmation"] = True
            draft_text = _format_draft_previews(draft_results)
            history.append(Content(role="assistant", parts=[Part.from_text(draft_text)]))
            if return_meta:
                return draft_text, history, meta
            return draft_text, history
