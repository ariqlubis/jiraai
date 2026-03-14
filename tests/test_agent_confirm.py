import unittest
from unittest.mock import patch
import sys
import types


def _install_vertexai_stub():
    vertexai = types.ModuleType("vertexai")

    def init(**_kwargs):
        return None

    vertexai.init = init

    gm = types.ModuleType("vertexai.generative_models")

    class GenerativeModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate_content(self, *_args, **_kwargs):
            raise RuntimeError("generate_content is not available in tests")

    class GenerationConfig:
        def __init__(self, *args, **kwargs):
            pass

    class Tool:
        def __init__(self, *args, **kwargs):
            pass

    class FunctionDeclaration:
        def __init__(self, *args, **kwargs):
            pass

    class Part:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def from_function_response(name, response):
            return {"name": name, "response": response}

        @staticmethod
        def from_text(text):
            return {"text": text}

    class Content:
        def __init__(self, *args, **kwargs):
            pass

    gm.GenerativeModel = GenerativeModel
    gm.GenerationConfig = GenerationConfig
    gm.Tool = Tool
    gm.FunctionDeclaration = FunctionDeclaration
    gm.Part = Part
    gm.Content = Content

    sys.modules.setdefault("vertexai", vertexai)
    sys.modules.setdefault("vertexai.generative_models", gm)


_install_vertexai_stub()


def _install_dotenv_stub():
    dotenv = types.ModuleType("dotenv")

    def load_dotenv(*_args, **_kwargs):
        return None

    dotenv.load_dotenv = load_dotenv
    sys.modules.setdefault("dotenv", dotenv)


def _install_loguru_stub():
    loguru = types.ModuleType("loguru")

    class _Logger:
        def info(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

    loguru.logger = _Logger()
    sys.modules.setdefault("loguru", loguru)


_install_dotenv_stub()
_install_loguru_stub()


def _install_atlassian_stub():
    atlassian = types.ModuleType("atlassian")

    class Jira:
        def __init__(self, *args, **kwargs):
            pass

    atlassian.Jira = Jira
    sys.modules.setdefault("atlassian", atlassian)


_install_atlassian_stub()

import agent.agent as agent_mod


class TestSystemPrompt(unittest.TestCase):
    def test_build_prompt_engineer_pro_includes_engineer_template_and_confirm_rules(self):
        prompt = agent_mod._build_system_prompt(mode="engineer", tone="pro")
        self.assertIn("Write safety (must follow):", prompt)
        self.assertIn("Engineer mode description template", prompt)
        self.assertNotIn("roast vague or lazy requests", prompt)

    def test_build_prompt_pm_snarky_includes_pm_template(self):
        prompt = agent_mod._build_system_prompt(mode="pm", tone="snarky")
        self.assertIn("Write safety (must follow):", prompt)
        self.assertIn("PM mode description template", prompt)
        self.assertIn("roast vague or lazy requests", prompt)


class TestDispatchConfirmationGate(unittest.TestCase):
    @patch("agent.agent.create_jira")
    def test_create_ticket_requires_confirmation_by_default(self, create_jira):
        args = {
            "project_key": "PROJ",
            "title": "Test ticket",
            "description": "Hello",
            "issue_type": "Task",
            "priority": "Medium",
        }
        result = agent_mod._dispatch("create_ticket", args, allow_writes=False)
        self.assertTrue(result.get("needs_confirmation"))
        self.assertEqual(result.get("action"), "create_ticket")
        create_jira.assert_not_called()

    @patch("agent.agent.create_jira")
    def test_create_ticket_confirm_true_but_writes_not_allowed_still_requires_confirmation(self, create_jira):
        args = {
            "project_key": "PROJ",
            "title": "Test ticket",
            "description": "Hello",
            "issue_type": "Task",
            "priority": "Medium",
            "confirm": True,
        }
        result = agent_mod._dispatch("create_ticket", args, allow_writes=False)
        self.assertTrue(result.get("needs_confirmation"))
        create_jira.assert_not_called()

    @patch("agent.agent.create_jira")
    def test_create_ticket_confirm_true_and_writes_allowed_creates_ticket(self, create_jira):
        create_jira.return_value = {"key": "PROJ-1", "message": "ok"}
        args = {
            "project_key": "PROJ",
            "title": "Test ticket",
            "description": "Hello",
            "issue_type": "Task",
            "priority": "Medium",
            "confirm": True,
        }
        result = agent_mod._dispatch("create_ticket", args, allow_writes=True)
        self.assertFalse(result.get("needs_confirmation", False))
        self.assertEqual(result.get("key"), "PROJ-1")
        create_jira.assert_called_once()

    @patch("agent.agent.update_ticket_fields")
    def test_update_ticket_requires_confirmation_by_default(self, update_ticket_fields):
        args = {
            "ticket_key": "PROJ-2",
            "description": "New desc",
        }
        result = agent_mod._dispatch("update_ticket", args, allow_writes=False)
        self.assertTrue(result.get("needs_confirmation"))
        self.assertEqual(result.get("action"), "update_ticket")
        update_ticket_fields.assert_not_called()

    @patch("agent.agent.update_ticket_fields")
    def test_update_ticket_confirm_true_and_writes_allowed_updates_ticket(self, update_ticket_fields):
        update_ticket_fields.return_value = {"success": True, "message": "updated"}
        args = {
            "ticket_key": "PROJ-2",
            "description": "New desc",
            "confirm": True,
        }
        result = agent_mod._dispatch("update_ticket", args, allow_writes=True)
        self.assertFalse(result.get("needs_confirmation", False))
        self.assertTrue(result.get("success"))
        update_ticket_fields.assert_called_once()

    @patch("agent.agent.create_subtasks")
    def test_create_subtasks_requires_confirmation_by_default(self, create_subtasks):
        args = {
            "parent_key": "PROJ-3",
            "subtasks": [{"title": "Sub 1", "description": "Desc"}],
        }
        result = agent_mod._dispatch("create_subtasks", args, allow_writes=False)
        self.assertTrue(result.get("needs_confirmation"))
        self.assertEqual(result.get("action"), "create_subtasks")
        create_subtasks.assert_not_called()

    @patch("agent.agent.create_subtasks")
    def test_create_subtasks_confirm_true_and_writes_allowed_creates(self, create_subtasks):
        create_subtasks.return_value = {"success": True, "subtasks": [{"key": "PROJ-4"}]}
        args = {
            "parent_key": "PROJ-3",
            "subtasks": [{"title": "Sub 1", "description": "Desc"}],
            "confirm": True,
        }
        result = agent_mod._dispatch("create_subtasks", args, allow_writes=True)
        self.assertFalse(result.get("needs_confirmation", False))
        self.assertTrue(result.get("success"))
        create_subtasks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
