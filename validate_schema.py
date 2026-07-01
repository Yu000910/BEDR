#!/usr/bin/env python3
"""Schema validation and verb distribution analysis for intent_action_entity.json.

Validates the output of s1_desc2graph.py against the expected graph schema
and reports the distribution of action verbs relative to the canonical
vocabulary defined in the system prompt.
"""
import json
import sys
from collections import Counter
from pathlib import Path

# Canonical recommended verb vocabulary from s1_desc2graph.py system prompt
CANONICAL_VERBS = {
    "search", "find", "collect", "query", "dump", "extract",
    "write", "drop", "save", "store", "encode", "encrypt",
    "execute", "run", "launch", "abuse", "leverage", "exploit",
    "modify", "replace", "patch", "hijack", "inject",
    "remove", "delete", "clear", "wipe",
}

# Nine entity categories from the system prompt
ENTITY_CATEGORIES = [
    "E1 Executable Files", "E2 Scripts/Code Blocks",
    "E3 System Commands/Switches", "E4 Device/Interface/Protocol",
    "E5 Registry Keys/Values", "E6 Credential Containers/Files",
    "E7 Network Resources", "E8 Memory/Kernel Objects",
    "E9 Logs/Artifact Files",
]

TARGET_FILE = "intent_action_entity.json"


def load_graphs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object, got {type(data).__name__}")
    return data


def validate_structure(graphs):
    """Validate the top-level schema of every graph entry."""
    errors = []
    for tech_id, g in graphs.items():
        if not isinstance(g, dict):
            errors.append(f"{tech_id}: not a dict, got {type(g).__name__}")
            continue
        for key in ("intent", "action_chain", "entities"):
            if key not in g:
                errors.append(f"{tech_id}: missing key '{key}'")
        if "action_chain" in g:
            if not isinstance(g["action_chain"], list):
                errors.append(f"{tech_id}: action_chain is not a list")
            else:
                for i, step in enumerate(g["action_chain"]):
                    if not isinstance(step, dict):
                        errors.append(f"{tech_id}.action_chain[{i}]: not a dict")
                    elif "verb" not in step or "tool" not in step:
                        errors.append(
                            f"{tech_id}.action_chain[{i}]: missing verb/tool"
                        )
        if "entities" in g and not isinstance(g["entities"], list):
            errors.append(f"{tech_id}: entities is not a list")
    return errors


def analyze_verbs(graphs):
    """Collect all verbs and classify them."""
    all_verbs = []
    for tech_id, g in graphs.items():
        for step in g.get("action_chain", []):
            if isinstance(step, dict) and "verb" in step:
                all_verbs.append(step["verb"].lower().strip())

    counter = Counter(all_verbs)
    in_vocab = {v for v in counter if v in CANONICAL_VERBS}
    out_vocab = {v for v in counter if v not in CANONICAL_VERBS}
    return counter, in_vocab, out_vocab


def main():
    path = Path(TARGET_FILE)
    if not path.exists():
        print(f"Error: '{TARGET_FILE}' not found in current directory.")
        sys.exit(1)

    graphs = load_graphs(path)
    print(f"Loaded {len(graphs)} technology entries from '{TARGET_FILE}'.\n")

    # ── 1. Structural validation ──────────────────────────────────
    errors = validate_structure(graphs)
    print("=" * 62)
    print("1. Structural Schema Validation")
    print("=" * 62)
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  • {e}")
    else:
        print("PASSED — all {:,} entries conform to the schema.".format(len(graphs)))

    # ── 2. Verb distribution ──────────────────────────────────────
    counter, in_vocab, out_vocab = analyze_verbs(graphs)
    total_verbs = sum(counter.values())

    print(f"\n{'=' * 62}")
    print("2. Action Verb Distribution")
    print("=" * 62)
    print(f"{'Total verb tokens':<40s} {total_verbs:>6d}")
    print(f"{'Unique verb types':<40s} {len(counter):>6d}")
    print(f"{'Types in canonical vocabulary':<40s} {len(in_vocab):>6d}")
    print(f"{'Types outside canonical vocabulary':<40s} {len(out_vocab):>6d}")
    print()

    # Token-level statistics
    in_count = sum(counter[v] for v in in_vocab)
    out_count = sum(counter[v] for v in out_vocab)
    print(f"{'Tokens in canonical vocabulary':<40s} {in_count:>6d}  ({in_count/total_verbs*100:5.1f}%)")
    print(f"{'Tokens outside canonical vocabulary':<40s} {out_count:>6d}  ({out_count/total_verbs*100:5.1f}%)")
    print()

    # Top-15 most frequent verbs
    print("Top-15 most frequent verbs (all):")
    for verb, count in counter.most_common(15):
        tag = " [canonical]" if verb in CANONICAL_VERBS else " [OOV]"
        print(f"  {verb:<20s} {count:>5d}{tag}")

    # OOV verb list (the reviewer's concern)
    print(f"\nOut-of-Vocabulary verbs ({len(out_vocab)} types):")
    for v in sorted(out_vocab):
        print(f"  {v:<20s} {counter[v]:>5d}")

    # ── 3. Summary statistics ─────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("3. Summary Statistics")
    print("=" * 62)
    # Count entries per graph
    empty_intents = sum(1 for g in graphs.values() if not g.get("intent", "").strip())
    empty_chains = sum(1 for g in graphs.values() if len(g.get("action_chain", [])) == 0)
    avg_chain_len = sum(len(g.get("action_chain", [])) for g in graphs.values()) / len(graphs)
    avg_entities = sum(len(g.get("entities", [])) for g in graphs.values()) / len(graphs)

    print(f"Entries with empty intent:       {empty_intents:>5d} / {len(graphs)}")
    print(f"Entries with empty action_chain:  {empty_chains:>5d} / {len(graphs)}")
    print(f"Average action_chain length:      {avg_chain_len:>6.2f}")
    print(f"Average entities count:           {avg_entities:>6.2f}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
