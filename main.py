import sys
import argparse
import os
from pathlib import Path
from shutil import get_terminal_size
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint
from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document

from agent.agent import run_agent

console = Console()
UI_MODE = os.getenv("JIRAAI_UI", "modern").strip().lower()
BASE_DIR = Path(__file__).resolve().parent

BANNER = """
     ██╗██╗██████╗  █████╗      █████╗ ██╗
     ██║██║██╔══██╗██╔══██╗    ██╔══██╗██║
     ██║██║██████╔╝███████║    ███████║██║
██   ██║██║██╔══██╗██╔══██║    ██╔══██║██║
╚█████╔╝██║██║  ██║██║  ██║    ██║  ██║██║
 ╚════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
"""


def print_banner():
    if UI_MODE == "classic":
        console.print(BANNER, style="bold cyan")
        console.print("  AI-powered Jira ticket generator\n", style="dim")
        return

    header = "[bold]JiraAI[/bold]  ·  AI-powered Jira ticket generator"
    console.print(Panel(header, border_style="cyan", padding=(1, 2)))


def print_response(text: str):
    """Render the agent's reply as Markdown."""
    console.print(Panel(Markdown(text), border_style="cyan", padding=(1, 2)))


def print_status(mode: str, tone: str):
    if UI_MODE == "classic":
        return
    status = f"[dim]Mode:[/dim] {mode}  ·  [dim]Tone:[/dim] {tone}  ·  [dim]Confirm:[/dim] on"
    console.print(Panel(status, border_style="bright_black", padding=(0, 2)))


def print_user_message(text: str):
    if UI_MODE == "classic":
        return
    console.print(Panel(text, border_style="magenta", title="You", padding=(1, 2)))

def _is_explicit_confirm(text: str) -> bool:
    return text.strip().lower() in {"y", "yes", "confirm", "apply", "approve"}


# ── Engineer mode ──────────────────────────────────────────────────────────────

class _AtPathCompleter(Completer):
    def __init__(self):
        self._path = PathCompleter(
            expanduser=True,
            get_paths=lambda: [str(BASE_DIR)],
        )

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text:
            return

        # Only complete when the current token starts with '@'
        last_space = max(text.rfind(" "), text.rfind("\t"), text.rfind("\n"))
        token = text[last_space + 1:] if last_space >= 0 else text
        if not token.startswith("@"):
            return

        token_path = token[1:]
        sub_doc = Document(text=token_path, cursor_position=len(token_path))
        for completion in self._path.get_completions(sub_doc, complete_event):
            yield Completion(
                completion.text,
                start_position=completion.start_position,
                display=completion.display,
                display_meta=completion.display_meta,
            )


_AT_PATH_COMPLETER = _AtPathCompleter()


def _prompt_with_path_complete(label: str) -> str:
    prompt_label = f"{label} " if label else ""
    use_box = UI_MODE == "modern"

    if not use_box:
        return prompt(
            prompt_label,
            completer=_AT_PATH_COMPLETER,
            complete_while_typing=False,
        ).strip()

    width = max(40, get_terminal_size((80, 20)).columns)
    console.print("")
    top = "┌" + "─" * (width - 2) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"
    console.print(top, markup=False, style="bright_black")
    if prompt_label:
        label_text = prompt_label.strip()
        if len(label_text) > width - 4:
            label_text = label_text[: width - 7] + "..."
        padding = " " * max(0, width - 4 - len(label_text))
        console.print(f"│ {label_text}{padding} │", markup=False, style="bright_black")

    user_input = prompt(
        "│ ",
        rprompt="│",
        completer=_AT_PATH_COMPLETER,
        complete_while_typing=False,
        bottom_toolbar=bottom,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "│ ",
    ).strip()
    console.print(bottom, markup=False, style="bright_black")
    console.print("")
    console.print("")
    return user_input


def engineer_mode():
    """
    Single-shot mode for engineers.
    Prompts for project key + a free-form description, then creates the ticket.
    """
    print_banner()
    print_status("engineer", os.getenv("JIRAAI_TONE", "pro"))
    console.print("[bold yellow]Engineer Mode[/bold yellow] — describe what needs doing, AI does the rest.\n")

    project_key = Prompt.ask("[bold]Project key[/bold] (e.g. PROJ)").strip().upper()
    if not project_key:
        console.print("[red]Project key is required.[/red]")
        sys.exit(1)

    console.print("\nDescribe the ticket (be as vague or precise as you want):")
    console.print("[dim]Press Enter twice to submit.[/dim]\n")

    lines = []
    try:
        while True:
            line = _prompt_with_path_complete("")
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
    except EOFError:
        pass

    user_input = "\n".join(lines).strip()
    if not user_input:
        console.print("[red]Nothing to work with. Aborting.[/red]")
        sys.exit(1)

    message = (
        f"Project key: {project_key}\n\n"
        f"Here's what needs a Jira ticket:\n{user_input}"
    )

    console.print("\n[dim]Thinking...[/dim]")
    try:
        if UI_MODE != "classic":
            print_user_message(message)
        tone = os.getenv("JIRAAI_TONE", "pro")
        reply, history, meta = run_agent(
            message,
            history=[],
            tone=tone,
            mode="engineer",
            allow_writes=False,
            return_meta=True,
        )
        print_response(reply)

        # Follow-up loop for draft edits + explicit confirmation.
        while not meta.get("write_performed"):
            followup = Prompt.ask("[bold yellow]Confirm/Edit[/bold yellow]").strip()
            if not followup:
                continue
            if followup.lower() in {"no", "n", "cancel", "exit", "quit"}:
                console.print("[dim]Cancelled.[/dim]")
                break

            reply, history, meta = run_agent(
                followup,
                history=history,
                tone=tone,
                mode="engineer",
                allow_writes=_is_explicit_confirm(followup),
                return_meta=True,
            )
            print_response(reply)
    except Exception as e:
        logger.error(e)
        console.print(f"[red]Agent error: {e}[/red]")
        sys.exit(1)


# ── PM (chat) mode ─────────────────────────────────────────────────────────────

def pm_mode():
    print_banner()
    print_status("pm", os.getenv("JIRAAI_TONE", "pro"))
    console.print("[bold magenta]PM Chat Mode[/bold magenta] — talk to the agent like a human.\n")
    console.print("[dim]Type 'exit' or 'quit' to leave. Type 'reset' to start a new conversation.[/dim]\n")

    history = []

    while True:
        try:
            user_input = _prompt_with_path_complete("jiraai ›")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower() == "reset":
            history = []
            console.print("[dim]Conversation reset.[/dim]\n")
            continue

        console.print("[dim]Thinking...[/dim]")
        try:
            allow_writes = _is_explicit_confirm(user_input)
            if UI_MODE != "classic":
                print_user_message(user_input)
            reply, history = run_agent(
                user_input,
                history=history,
                tone=os.getenv("JIRAAI_TONE", "pro"),
                mode="pm",
                allow_writes=allow_writes,
            )
            print_response(reply)
        except Exception as e:
            logger.error(e)
            console.print(f"[red]Agent error: {e}[/red]")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="jiraai",
        description="JiraAI — AI-powered Jira ticket manager",
    )
    parser.add_argument(
        "mode",
        choices=["engineer", "pm"],
        help=(
            "'engineer' for one-shot ticket creation, "
            "'pm' for interactive chat mode"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: WARNING)",
    )
    parser.add_argument(
        "--tone",
        default=os.getenv("JIRAAI_TONE", "pro"),
        choices=["pro", "snarky"],
        help="Response style (default: pro). Also configurable via JIRAAI_TONE.",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    os.environ["JIRAAI_TONE"] = args.tone

    if args.mode == "engineer":
        engineer_mode()
    elif args.mode == "pm":
        pm_mode()


if __name__ == "__main__":
    main()
