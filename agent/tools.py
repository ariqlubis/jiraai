import os
from typing import Optional, Any
from atlassian import Jira
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

def get_jira_client() -> Jira:
    return Jira(
        url=os.getenv("JIRA_URL"),
        username=os.getenv("JIRA_USERNAME"),
        password=os.getenv("JIRA_PASSWORD"),
        cloud=True
    )

def create_jira(
    project_key: str,
    title: str,
    description: str,
    issue_type: str,
    priority: str = 'Task',
    assignee_email: Optional[str] = None
) -> dict:
    jira = get_jira_client()

    fields: dict[str, Any] = {
        'project': {'key': project_key},
        'summary': title,
        'description': description,
        'issuetype': {'name': issue_type},
        'priority': {'name': priority}
    }

    if assignee_email:
        user = jira.user_find_by_user_string(query=assignee_email, max_results=1)
        if user:
            fields['assignee'] = {
                "accountId": user[0]['accountId']
            }

    result = jira.issue_create(fields=fields)
    base_url = os.getenv("JIRA_URL").rstrip("/")
    return {
        "key": result['key'],
        "id": result['id'],
        "url": f"{base_url}/browse/{result['key']}",
        "message": f"Ticket {result['key']} created successfully"
    }

def get_ticket(ticket_key: str) -> dict:
    jira = get_jira_client()
    issue = jira.issue(ticket_key)
    fields = issue['fields']

    return {
        "key": issue["key"],
        "title": fields.get("summary", ""),
        "description": fields.get("description", ""),
        "status": fields.get("status", {}).get("name", ""),
        "type": fields.get("issuetype", {}).get("name", ""),
        "priority": fields.get("priority", {}).get("name", ""),
        "assignee": fields.get("assignee", {}).get("displayName", "Unassigned") if fields.get("assignee") else "Unassigned",
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
        fields['summary'] = title
    if description:
        fields['description'] = description
    if priority:
        fields['priority'] = {'name': priority}
    if assignee_email:
        user = jira.user_find_by_user_string(query=assignee_email, max_results=1)
        if user:
            fields['assignee'] = {
                "accountId": user[0]['accountId']
            }

    if not fields:
        return {
            "success": False,
            "message": "No fields to update"
        }

    jira.issue_update(issue=ticket_key, fields=fields)
    return {
        "success": True,
        "message": f"Ticket {ticket_key} updated successfully"
    }

def list_tickets(
    project_key: str,
    status: Optional[str] = None,
    assignee_email: Optional[str] = None,
    max_results: int = 10
) -> dict:
    jira = get_jira_client()
    jql = f"project = '{project_key}'"
    if status:
        jql += f" AND status = '{status}'"
    if assignee_email:
        jql += f" AND assignee = '{assignee_email}'"
    jql += " ORDER BY created DESC"


    results = jira.jql(jql=jql, max_results=max_results)
    tickets = []

    for issue in results.get("issues", []):
        fields = issue['fields']
        tickets.append({
            "key": issue["key"],
            "title": fields.get("summary", ""),
            "status": fields.get("status", {}).get("name", ""),
            "type": fields.get("issuetype", {}).get("name", ""),
            "priority": fields.get("priority", {}).get("name", ""),
            "assignee": fields.get("assignee", {}).get("displayName", "Unassigned") if fields.get("assignee") else "Unassigned",
        })

    return {
        "total": results.get("total", 0),
        "tickets": tickets
    }