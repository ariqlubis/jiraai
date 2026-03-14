import os
from functools import lru_cache
from urllib.parse import urlparse
from typing import Optional, Any, Union
import re
from atlassian import Jira
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is not set")
    return value


@lru_cache(maxsize=1)
def _get_base_url() -> str:
    base_url = _require_env("JIRA_URL").rstrip("/")
    parsed = urlparse(base_url)
    if not parsed.scheme:
        raise ValueError("JIRA_URL must include a scheme such as https://")
    return base_url


_ADF_ENABLED: Optional[bool] = None


def _resolve_adf_preference() -> bool:
    override = os.getenv("JIRA_USE_ADF")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}

    base_url = _get_base_url()
    return ".atlassian.net" in base_url.lower()


def _use_adf_description() -> bool:
    global _ADF_ENABLED
    if _ADF_ENABLED is None:
        _ADF_ENABLED = _resolve_adf_preference()
    return bool(_ADF_ENABLED)


def _disable_adf_description():
    global _ADF_ENABLED
    _ADF_ENABLED = False


def _paragraph_node(text: str) -> dict:
    if not text:
        return {"type": "paragraph", "content": []}
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}


def _list_item_node(text: str) -> dict:
    return {"type": "listItem", "content": [_paragraph_node(text)]}


def _markdown_to_adf(description: str) -> dict:
    lines = description.splitlines()
    content: list[dict[str, Any]] = []
    current_list: list[dict[str, Any]] = []
    current_list_type: Optional[str] = None

    def flush_list() -> None:
        nonlocal current_list, current_list_type
        if current_list:
            content.append({"type": current_list_type, "content": current_list})
            current_list = []
            current_list_type = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_list()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_list()
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            content.append({
                "type": "heading",
                "attrs": {"level": level},
                "content": [{"type": "text", "text": text}] if text else [],
            })
            continue

        bullet_match = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet_match:
            if current_list_type not in (None, "bulletList"):
                flush_list()
            current_list_type = "bulletList"
            current_list.append(_list_item_node(bullet_match.group(1).strip()))
            continue

        ordered_match = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if ordered_match:
            if current_list_type not in (None, "orderedList"):
                flush_list()
            current_list_type = "orderedList"
            current_list.append(_list_item_node(ordered_match.group(1).strip()))
            continue

        flush_list()
        content.append(_paragraph_node(line))

    flush_list()
    if not content:
        content = [_paragraph_node("")]

    return {
        "type": "doc",
        "version": 1,
        "content": content,
    }


def _format_description(description: str) -> Union[str, dict]:
    """
    Convert a plain-text description into the shape expected by the Jira API.
    Atlassian Cloud requires Atlassian Document Format; Server/Data Center still accept strings.
    """
    if not description:
        return ""

    if not _use_adf_description():
        return description

    return _markdown_to_adf(description)


def _extract_text_from_node(node: dict, ordered_index: Optional[int] = None) -> list[str]:
    node_type = node.get("type")
    if node_type in {"paragraph", "heading"}:
        parts = []
        for child in node.get("content", []):
            if child.get("type") == "text":
                parts.append(child.get("text", ""))
        text = "".join(parts).strip()
        return [text] if text else []

    if node_type in {"bulletList", "orderedList"}:
        lines = []
        items = node.get("content", [])
        for idx, item in enumerate(items, start=1):
            item_lines: list[str] = []
            for child in item.get("content", []):
                item_lines.extend(_extract_text_from_node(child))
            item_text = " ".join([part for part in item_lines if part]).strip()
            if not item_text:
                continue
            if node_type == "orderedList":
                lines.append(f"{idx}. {item_text}")
            else:
                lines.append(f"- {item_text}")
        return lines

    if node_type == "listItem":
        lines = []
        for child in node.get("content", []):
            lines.extend(_extract_text_from_node(child, ordered_index=ordered_index))
        return lines

    return []


def _parse_description(raw_description: Any) -> str:
    """
    Best-effort conversion of Atlassian Document Format back into readable text.
    """
    if isinstance(raw_description, str):
        return raw_description

    if not isinstance(raw_description, dict) or raw_description.get("type") != "doc":
        return ""

    lines: list[str] = []
    for node in raw_description.get("content", []):
        lines.extend(_extract_text_from_node(node))

    return "\n".join([line for line in lines if line is not None])


def _project_key_from_ticket_key(ticket_key: str) -> str:
    if "-" not in ticket_key:
        raise ValueError("Invalid Jira ticket key")
    return ticket_key.split("-", 1)[0]

def _should_retry_with_plain_text(error: Exception) -> bool:
    message = str(error).lower()
    hints = [
        "must be a string",
        "should be a string",
        "value needs to be string",
    ]
    return any(hint in message for hint in hints)


def _apply_description_for_retry(
    fields: dict[str, Any],
    description: Optional[str],
) -> Optional[str]:
    """
    Attach a formatted description to the payload and return the original
    text so we can fall back to plain text when needed.
    """
    if description is None:
        return None
    fields["description"] = _format_description(description)
    return description or ""


def get_jira_client() -> Jira:
    base_url = _get_base_url()
    username = _require_env("JIRA_USERNAME")
    password = _require_env("JIRA_PASSWORD")
    return Jira(
        url=base_url,
        username=username,
        password=password,
        cloud=True,
        api_version="3",
    )


def create_jira(
    project_key: str,
    title: str,
    description: str,
    issue_type: str,
    priority: str = "Medium",
    assignee_email: Optional[str] = None,
    subtasks: Optional[list[dict[str, Any]]] = None,
) -> dict:
    jira = get_jira_client()

    fields: dict[str, Any] = {
        "project": {"key": project_key},
        "summary": title,
        "issuetype": {"name": issue_type},
        "priority": {"name": priority},
    }
    original_description = _apply_description_for_retry(fields, description)

    if assignee_email:
        user = jira.user_find_by_user_string(query=assignee_email, limit=1)
        if user:
            fields["assignee"] = {"accountId": user[0]["accountId"]}

    try:
        result = jira.issue_create(fields=fields)
    except Exception as exc:
        if original_description is not None and _use_adf_description() and _should_retry_with_plain_text(exc):
            logger.warning("Jira rejected Atlassian Document Format description, retrying with plain text")
            _disable_adf_description()
            fields["description"] = original_description
            result = jira.issue_create(fields=fields)
        else:
            raise
    base_url = _get_base_url()
    response = {
        "key": result["key"],
        "id": result["id"],
        "url": f"{base_url}/browse/{result['key']}",
        "message": f"Ticket {result['key']} created successfully",
    }

    if subtasks:
        subtask_result = create_subtasks(result["key"], subtasks)
        response["subtasks"] = subtask_result.get("subtasks", [])
        response["subtasks_message"] = subtask_result.get("message")

    return response


def get_ticket(ticket_key: str) -> dict:
    jira = get_jira_client()
    issue = jira.issue(ticket_key)
    fields = issue["fields"]

    return {
        "key": issue["key"],
        "title": fields.get("summary", ""),
        "description": _parse_description(fields.get("description")),
        "status": fields.get("status", {}).get("name", ""),
        "type": fields.get("issuetype", {}).get("name", ""),
        "priority": fields.get("priority", {}).get("name", ""),
        "assignee": (
            fields.get("assignee", {}).get("displayName", "Unassigned")
            if fields.get("assignee")
            else "Unassigned"
        ),
        "reporter": fields.get("reporter", {}).get("displayName", ""),
        "created": fields.get("created", ""),
        "updated": fields.get("updated", ""),
    }


def update_ticket_fields(
    ticket_key: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    priority: Optional[str] = None,
    assignee_email: Optional[str] = None,
) -> dict:
    jira = get_jira_client()
    fields = {}

    if title:
        fields["summary"] = title
    original_description = _apply_description_for_retry(fields, description)
    if priority:
        fields["priority"] = {"name": priority}
    if assignee_email:
        user = jira.user_find_by_user_string(query=assignee_email, limit=1)
        if user:
            fields["assignee"] = {"accountId": user[0]["accountId"]}

    if not fields:
        return {"success": False, "message": "No fields to update"}

    try:
        jira.issue_update(ticket_key, fields=fields)
    except Exception as exc:
        if original_description is not None and _use_adf_description() and _should_retry_with_plain_text(exc):
            logger.warning("Jira rejected Atlassian Document Format description, retrying with plain text")
            _disable_adf_description()
            fields["description"] = original_description
            jira.issue_update(ticket_key, fields=fields)
        else:
            raise
    return {"success": True, "message": f"Ticket {ticket_key} updated successfully"}


def list_tickets(
    project_key: str,
    status: Optional[str] = None,
    assignee_email: Optional[str] = None,
    max_results: int = 10,
) -> dict:
    jira = get_jira_client()
    jql = f"project = '{project_key}'"
    if status:
        jql += f" AND status = '{status}'"
    if assignee_email:
        jql += f" AND assignee = '{assignee_email}'"
    jql += " ORDER BY created DESC"

    results = jira.jql(jql=jql, limit=max_results)
    tickets = []

    for issue in results.get("issues", []):
        fields = issue["fields"]
        tickets.append({
            "key": issue["key"],
            "title": fields.get("summary", ""),
            "status": fields.get("status", {}).get("name", ""),
            "type": fields.get("issuetype", {}).get("name", ""),
            "priority": fields.get("priority", {}).get("name", ""),
            "assignee": (
                fields.get("assignee", {}).get("displayName", "Unassigned")
                if fields.get("assignee")
                else "Unassigned"
            ),
        })

    return {"total": results.get("total", 0), "tickets": tickets}


def create_subtasks(
    parent_key: str,
    subtasks: list[dict[str, Any]],
) -> dict:
    if not subtasks:
        return {"success": False, "message": "No subtasks provided"}

    jira = get_jira_client()
    project_key = _project_key_from_ticket_key(parent_key)
    base_url = _get_base_url()
    created = []

    for subtask in subtasks:
        title = subtask.get("title")
        description = subtask.get("description", "")
        priority = subtask.get("priority", "Medium")
        assignee_email = subtask.get("assignee_email")

        if not title:
            raise ValueError("Subtask title is required")

        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "parent": {"key": parent_key},
            "summary": title,
            "issuetype": {"name": "Sub-task"},
            "priority": {"name": priority},
        }
        original_description = _apply_description_for_retry(fields, description)

        if assignee_email:
            user = jira.user_find_by_user_string(query=assignee_email, limit=1)
            if user:
                fields["assignee"] = {"accountId": user[0]["accountId"]}

        try:
            result = jira.issue_create(fields=fields)
        except Exception as exc:
            if original_description is not None and _use_adf_description() and _should_retry_with_plain_text(exc):
                logger.warning("Jira rejected Atlassian Document Format description, retrying with plain text")
                _disable_adf_description()
                fields["description"] = original_description
                result = jira.issue_create(fields=fields)
            else:
                raise

        created.append({
            "key": result["key"],
            "id": result["id"],
            "url": f"{base_url}/browse/{result['key']}",
        })

    return {
        "success": True,
        "message": f"Created {len(created)} subtasks under {parent_key}",
        "subtasks": created,
    }
