import vertexai
import os
import json
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    FunctionDeclaration,
    Part
)

from dotenv import load_dotenv
from agent.tools import (
    create_jira,
    get_ticket,
    update_ticket_fields,
    list_tickets
)

from agent.classifier import predict_type

load_dotenv()

vertexai.init(
    project_id=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION")
)

TOOLS = Tool(function_declarations=[
    FunctionDeclaration(
        name='predict_ticket_type',
        description="Use ML model to predict ticket type (Bug/Feature/Task). Always call this before creating a ticket.",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Ticket title"
                },
                "description": {
                    "type": "string",
                    "description": "Ticket description (optional)"
                }
            },
            "required": ["title"]
        }
    ),
])

