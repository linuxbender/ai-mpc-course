import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

notes_mcp = FastMCP(name="Notes MCP Server")

NOTES_FILE = Path(__file__).parent / "data/notes.json"

def load_notes() -> dict:
    notes_file = NOTES_FILE
    if notes_file.exists():
        return json.loads(notes_file.read_text())
    return {}

@notes_mcp.tool()
def save_notes(notes: dict):
    NOTES_FILE.write_text(json.dumps(notes, indent=2))

@notes_mcp.tool()
def add_note(name: str, content: str) -> str:
    notes = load_notes()
    notes[name] = content
    save_notes(notes)
    return f"the Note {name} was added"

@notes_mcp.tool()
def delete_note(name: str) -> str:
    notes = load_notes()
    if name in notes:
        del notes[name]
        save_notes(notes)
        return f"the Note {name} was deleted"

    return f"the Note {name} was not found"

@notes_mcp.tool()
def get_note(name: str) -> str:
    notes = load_notes()
    if name in notes:
        return notes[name]
    return f"the Note {name} was not found"

@notes_mcp.tool()
def list_notes() -> str:
    notes = load_notes()
    if not notes:
        return f"the Notes are empty"
    return f"Notes: {', '.join(notes.keys())} notes"


@notes_mcp.resource("resource://{name}")
def get_note_resource(name: str) -> str:
    notes = load_notes()
    if name in notes:
        return notes[name]
    return f"the Note {name} was not found"

@notes_mcp.prompt()
def summarize_notes(name: str) -> str:
    notes = load_notes()
    if name not in notes:
        return f"the Note {name} was not found"
    return f"""
    Here is a note titled '{name}':
    {notes[name]}
    Please summarize it in a concise manner. Keep the summary 100 words or less.
    """

if __name__ == "__main__":
    notes_mcp.run()
