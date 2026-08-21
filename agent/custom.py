"""
Coding agent — pure LangGraph module, no web framework.

Tools:
  - read_file, list_directory, grep_code   -> read-only, no approval needed
  - write_file, run_command                -> PAUSE via interrupt() and wait
                                               for human approval before acting

Everything is confined to SANDBOX_ROOT so the agent can never touch files
outside the practice folder, regardless of what path it's given.

Requires:
    pip install langgraph langchain-ollama --break-system-packages

Make sure Ollama is running and you've pulled a tool-calling model:
    ollama serve
    ollama pull llama3.1
"""

import subprocess
from pathlib import Path

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# SANDBOX — every path the agent touches gets resolved against this and
# checked that it doesn't escape it (blocks "../../etc/passwd" style paths).
# ---------------------------------------------------------------------------
SANDBOX_ROOT = Path(__file__).parent / "sandbox"
SANDBOX_ROOT.mkdir(exist_ok=True)

def _resolve(path: str) -> Path:
    full = (SANDBOX_ROOT / path).resolve()
    if SANDBOX_ROOT.resolve() not in full.parents and full != SANDBOX_ROOT.resolve():
        raise ValueError(f"Path '{path}' escapes the sandbox — refused.")
    return full


# ---------------------------------------------------------------------------
# READ-ONLY TOOLS — no approval needed, safe to let the agent use freely
# ---------------------------------------------------------------------------
@tool
def read_file(path: str) -> str:
    """Read the full contents of a file in the sandbox. Path is relative
    to the sandbox root, e.g. 'example.py' or 'src/utils.py'."""
    try:
        full = _resolve(path)
        if not full.exists():
            return f"File not found: {path}"
        return full.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """List files and folders inside the given directory in the sandbox
    (relative path, default is the sandbox root)."""
    try:
        full = _resolve(path)
        if not full.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in full.iterdir())
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Error listing {path}: {e}"


@tool
def grep_code(pattern: str, path: str = ".") -> str:
    """Search for a text pattern across files in the sandbox. Returns
    matching lines with their file and line number."""
    try:
        full = _resolve(path)
        matches = []
        for file in full.rglob("*"):
            if file.is_file():
                try:
                    for i, line in enumerate(file.read_text().splitlines(), 1):
                        if pattern in line:
                            rel = file.relative_to(SANDBOX_ROOT)
                            matches.append(f"{rel}:{i}: {line.strip()}")
                except UnicodeDecodeError:
                    continue  # skip binary files
        if not matches:
            return f"No matches for '{pattern}'"
        return "\n".join(matches[:50])  # cap output
    except Exception as e:
        return f"Error searching: {e}"


# ---------------------------------------------------------------------------
# WRITE / EXECUTE TOOLS — these call interrupt() and PAUSE the whole graph
# until a human approves or rejects, before doing anything real.
# ---------------------------------------------------------------------------
@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the sandbox, creating or overwriting it.
    Requires human approval before executing."""
    decision = interrupt({
        "type": "approval",
        "action": "write_file",
        "path": path,
        "preview": content[:800],
    })
    if decision != "approve":
        return f"Write to '{path}' was rejected by the user."

    try:
        full = _resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return f"Wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Make a targeted edit to an existing file: find an exact snippet of
    text (old_text) and replace it with new_text. Safer than write_file for
    modifying existing code, since it only touches the specific lines that
    need to change. old_text must match exactly (including whitespace) and
    must appear exactly once in the file. Requires human approval."""
    try:
        full = _resolve(path)
        if not full.exists():
            return f"File not found: {path}"
        content = full.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"

    count = content.count(old_text)
    if count == 0:
        return f"old_text not found in {path} — no changes made. Re-check the exact text (whitespace matters)."
    if count > 1:
        return f"old_text appears {count} times in {path} — must be unique. Include more surrounding context to disambiguate."

    diff_preview = f"- {old_text}\n+ {new_text}"
    decision = interrupt({
        "type": "approval",
        "action": "edit_file",
        "path": path,
        "diff": diff_preview,
    })
    if decision != "approve":
        return f"Edit to '{path}' was rejected by the user."

    try:
        new_content = content.replace(old_text, new_text)
        full.write_text(new_content)
        return f"Edited {path}: replaced {len(old_text)} chars with {len(new_text)} chars."
    except Exception as e:
        return f"Error writing {path}: {e}"


@tool
def git_status() -> str:
    """Show the current git status of the sandbox repo (modified/untracked files)."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"], cwd=SANDBOX_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or "Working tree clean — no changes."
    except Exception as e:
        return f"Error running git status: {e}"


@tool
def git_diff() -> str:
    """Show the current uncommitted changes (git diff) in the sandbox repo."""
    try:
        result = subprocess.run(
            ["git", "diff"], cwd=SANDBOX_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or "No uncommitted changes."
    except Exception as e:
        return f"Error running git diff: {e}"


@tool
def git_commit(message: str) -> str:
    """Stage all changes and commit them to the sandbox's local git repo
    with the given commit message. Requires human approval."""
    decision = interrupt({
        "type": "approval",
        "action": "git_commit",
        "message": message,
    })
    if decision != "approve":
        return "Commit was rejected by the user."

    try:
        subprocess.run(["git", "add", "-A"], cwd=SANDBOX_ROOT, timeout=10, check=True)
        result = subprocess.run(
            ["git", "commit", "-m", message], cwd=SANDBOX_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return f"Nothing to commit, or error: {result.stdout}{result.stderr}"
        return result.stdout.strip()
    except Exception as e:
        return f"Error committing: {e}"


@tool
def run_command(command: str) -> str:
    """Run a shell command inside the sandbox directory (e.g. running
    tests: 'python -m pytest'). Requires human approval before executing."""
    decision = interrupt({
        "type": "approval",
        "action": "run_command",
        "command": command,
    })
    if decision != "approve":
        return f"Command '{command}' was rejected by the user."

    try:
        result = subprocess.run(
            command, shell=True, cwd=SANDBOX_ROOT,
            capture_output=True, text=True, timeout=15,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:2000] if output else f"(command exited {result.returncode}, no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 15 seconds."
    except Exception as e:
        return f"Error running command: {e}"


TOOLS = [
    read_file, list_directory, grep_code,
    write_file, edit_file, run_command,
    git_status, git_diff, git_commit,
]

# ---------------------------------------------------------------------------
# AGENT + GRAPH
# ---------------------------------------------------------------------------
import re
import uuid

llm = ChatOllama(model="llama3", temperature=0)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a careful coding assistant working inside a sandboxed practice "
    "folder that is also a local git repo. Use read_file, list_directory, "
    "grep_code, git_status, and git_diff freely to understand the code "
    "before making changes. Prefer edit_file over write_file when modifying "
    "an existing file — it makes a precise, reviewable change instead of "
    "rewriting the whole file. Use write_file only for brand new files. "
    "After making changes, consider running tests with run_command to "
    "verify they work before committing. Explain briefly what you're about "
    "to do and why before any write_file, edit_file, run_command, or "
    "git_commit call — the user will be asked to approve it before it "
    "actually happens.\n\n"
    "Because native tool calling is not supported, you must use a custom XML "
    "format to call tools. When you want to call one or more tools, write them "
    "using the <tool_call> tag. Do not include any other markdown code block formats around the XML tag.\n\n"
    "Format:\n"
    "<tool_call name=\"tool_name\">\n"
    "  <arg_name>arg_value</arg_name>\n"
    "</tool_call>\n\n"
    "Here are the available tools and their arguments:\n"
    "1. read_file(path: str)\n"
    "   Example:\n"
    "   <tool_call name=\"read_file\">\n"
    "     <path>hello.py</path>\n"
    "   </tool_call>\n\n"
    "2. list_directory(path: str = \".\")\n"
    "   Example:\n"
    "   <tool_call name=\"list_directory\">\n"
    "     <path>.</path>\n"
    "   </tool_call>\n\n"
    "3. grep_code(pattern: str, path: str = \".\")\n"
    "   Example:\n"
    "   <tool_call name=\"grep_code\">\n"
    "     <pattern>def calculate</pattern>\n"
    "     <path>src</path>\n"
    "   </tool_call>\n\n"
    "4. write_file(path: str, content: str)\n"
    "   Example:\n"
    "   <tool_call name=\"write_file\">\n"
    "     <path>test.txt</path>\n"
    "     <content>hello world</content>\n"
    "   </tool_call>\n\n"
    "5. edit_file(path: str, old_text: str, new_text: str)\n"
    "   Example:\n"
    "   <tool_call name=\"edit_file\">\n"
    "     <path>src/main.py</path>\n"
    "     <old_text>print(\"hello\")</old_text>\n"
    "     <new_text>print(\"hello world\")</new_text>\n"
    "   </tool_call>\n\n"
    "6. git_status()\n"
    "   Example:\n"
    "   <tool_call name=\"git_status\">\n"
    "   </tool_call>\n\n"
    "7. git_diff()\n"
    "   Example:\n"
    "   <tool_call name=\"git_diff\">\n"
    "   </tool_call>\n\n"
    "8. git_commit(message: str)\n"
    "   Example:\n"
    "   <tool_call name=\"git_commit\">\n"
    "     <message>initial commit</message>\n"
    "   </tool_call>\n\n"
    "9. run_command(command: str)\n"
    "   Example:\n"
    "   <tool_call name=\"run_command\">\n"
    "     <command>python -m pytest</command>\n"
    "   </tool_call>"
))

def parse_xml_tool_calls(text: str) -> list[dict]:
    pattern = re.compile(r'<tool_call\s+name=["\']([^"\']+)["\']\s*>(.*?)</tool_call>', re.DOTALL)
    tool_calls = []
    for match in pattern.finditer(text):
        tool_name = match.group(1)
        args_text = match.group(2)
        
        args = {}
        arg_pattern = re.compile(r'<([^>]+)>(.*?)</\1>', re.DOTALL)
        for arg_match in arg_pattern.finditer(args_text):
            arg_name = arg_match.group(1)
            arg_value = arg_match.group(2)
            args[arg_name] = arg_value.strip()
            
        tool_calls.append({
            'name': tool_name,
            'args': args,
            'id': f"call_{uuid.uuid4().hex[:12]}",
            'type': 'tool_call'
        })
    return tool_calls

def agent(state: MessagesState) -> dict:
    response = llm.invoke([SYSTEM_PROMPT] + state["messages"])
    tool_calls = parse_xml_tool_calls(response.content)
    if tool_calls:
        new_response = AIMessage(
            content=response.content,
            tool_calls=tool_calls,
            id=response.id,
            response_metadata=response.response_metadata,
            additional_kwargs=response.additional_kwargs
        )
        return {"messages": [new_response]}
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

# interrupt() REQUIRES a checkpointer — without one there's no saved state
# to pause and resume from.
checkpointer = InMemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Manual test — run this file directly to see the interrupt in action
# without any web server involved.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "manual-test"}}

    result = app_graph.invoke(
        {"messages": [("user", "Create a file called yafet.txt with the text 'Hello, yafet!'")]},
        config=config,
    )

    state = app_graph.get_state(config)
    if state.next:
        # Graph is paused waiting for approval
        pending = state.tasks[0].interrupts[0].value
        print("--- APPROVAL NEEDED ---")
        print(pending)
        decision = input("Approve? (yes/no): ").strip().lower()
        result = app_graph.invoke(
            Command(resume="approve" if decision == "yes" else "reject"),
            config=config,
        )

    print("\n--- FINAL ---")
    print(result["messages"][-1].content)
