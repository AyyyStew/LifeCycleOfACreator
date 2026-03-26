"""
Joins soft-wrapped lines within markdown cells in a jupytext percent-format .py file.
Preserves headings, list items, blank lines, and code blocks.
Usage: python fix_markdown_wrapping.py era_topic_analysis.py
"""
import re
import sys

def is_special(text):
    """Lines that should never be merged with the next line."""
    t = text.strip()
    return (
        t.startswith("#")   # heading
        or t.startswith("-") or t.startswith("*")  # list
        or t.startswith(">")   # blockquote
        or t.startswith("```") # code fence
        or t.startswith("|")   # table
        or re.match(r"^\d+\.", t)  # numbered list
        or t in ("---", "===", "***")  # horizontal rule
        or t == ""  # blank
    )

def fix_file(path):
    with open(path) as f:
        lines = f.readlines()

    out = []
    in_markdown = False
    buffer = []  # accumulates a run of wrappable lines

    def flush(buf):
        """Join a run of wrappable lines into one."""
        if buf:
            joined = "# " + " ".join(l[2:].rstrip("\n") for l in buf) + "\n"
            out.append(joined)
            buf.clear()

    for line in lines:
        stripped = line.rstrip("\n")

        # Cell boundary
        if stripped.startswith("# %%"):
            flush(buffer)
            in_markdown = stripped.startswith("# %% [markdown]")
            out.append(line)
            continue

        if not in_markdown:
            out.append(line)
            continue

        # Inside a markdown cell
        # Blank comment line → paragraph break, flush buffer
        if stripped == "#":
            flush(buffer)
            out.append(line)
            continue

        # Lines that aren't comments at all (shouldn't exist in markdown cells, but be safe)
        if not stripped.startswith("# "):
            flush(buffer)
            out.append(line)
            continue

        content = stripped[2:]  # text after "# "

        if is_special(content):
            flush(buffer)
            out.append(line)
        else:
            buffer.append(line)

    flush(buffer)

    with open(path, "w") as f:
        f.writelines(out)

    print(f"Done: {path}")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        fix_file(path)
