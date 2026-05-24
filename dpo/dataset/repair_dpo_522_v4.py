from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_V2 = ROOT / "DPO_522_prompt_a_and_prompt_b_V2.jsonl"
SOURCE_V3 = ROOT / "DPO_522_prompt_a_and_prompt_b_V3_repaired.jsonl"
OUTPUT = ROOT / "DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl"
MANIFEST = ROOT / "DPO_522_prompt_a_and_prompt_b_V4_repair_manifest.jsonl"
REPORT = ROOT / "DPO_522_prompt_a_and_prompt_b_V4_validation_report.json"


SCAFFOLD_RE = re.compile(
    r"Simple triage|One honest read|Skip the label for a second|If I had to bet a lunch|"
    r"Treat it like a leak you need to trace|Name the scene, not the mood|Let us not wallpaper this|"
    r"Okay, tight scope|I want one concrete lever, not vibes|"
    r"Before we spend money on the wrong fix|"
    r"direct problem|vague start|life/hobby|"
    r"posture, surface, or something you inhale or swallow|"
    r"Desk, bed, kitchen, or on the move|"
    r"What is the one thing that already failed you|"
    r"When does it hit hardest, time of day and position|"
    r"What is the pattern: constant, spikes, or only after one activity"
)

MALFORMED_REPLACEMENTS = [
    ("noise. accordion folder divider does", "noise. An accordion folder divider does"),
    ("desk. adjustable laptop riser lifts", "desk. An adjustable laptop riser lifts"),
    ("umbrellas. cantilever patio umbrella solves", "umbrellas. A cantilever patio umbrella solves"),
    ("surfaces. produce humidity saver pouch is", "surfaces. A produce humidity saver pouch is"),
    ("while? felt ukulele pick set gives", "while? A felt ukulele pick set gives"),
    ("friction. timed-release phone enclosure moves", "friction. A timed-release phone enclosure moves"),
    ("meal. ginger turmeric digestive chew cube is", "meal. A ginger turmeric digestive chew cube is"),
    ("everything. countertop water dispenser jug gives", "everything. A countertop water dispenser jug gives"),
    ("press. cocktail muddler is", "press. A cocktail muddler is"),
    ("vise. battery-heated handwarmer muff is", "vise. A battery-heated handwarmer muff is"),
    ("lottery. fermentation weight set is", "lottery. A fermentation weight set is"),
    ("volume. citrus-infused aromatherapy vial is", "volume. A citrus-infused aromatherapy vial is"),
    ("exposures. camera lens hood blocks", "exposures. A camera lens hood blocks"),
    ("quiet. guided body-scan audio bead pendant is", "quiet. A guided body-scan audio bead pendant is"),
    ("real. heavy-duty ratchet tie-down turns", "real. A heavy-duty ratchet tie-down turns"),
    ("sandwich. bone conduction sleep speaker sits", "sandwich. A bone conduction sleep speaker sits"),
    ("hairs. boar-bristle scalp brush used", "hairs. A boar-bristle scalp brush used"),
    ("barrier that does not depend on a single fold sitting on damp tile. dry roll pouch with", "barrier that does not depend on a single fold sitting on damp tile. A dry roll pouch with"),
    ("moves. color-coded index-card wallet set gives", "moves. A color-coded index-card wallet set gives"),
    ("lot? a sleep position trainer strap", "lot? A sleep position trainer strap"),
    ("optics. page magnifier gives", "optics. A page magnifier gives"),
    ("guts. saddle stitch stapler has", "guts. A saddle stitch stapler has"),
    ("second? a pocket-size breath anchor pendant", "second? A pocket-size breath anchor pendant"),
    ("birthday crown. heavy-gauge galvanized tomato cages", "birthday crown. A set of heavy-gauge galvanized tomato cages"),
    ("kid grease. leather conditioning wipe canister is", "kid grease. A leather conditioning wipe canister is"),
    ("periphery. ring-halo ambient glow disc behind", "periphery. A ring-halo ambient glow disc behind"),
    ("visually. darning thread sampler gives", "visually. A darning thread sampler gives"),
]


TARGETED_MESSAGE_REPAIRS: dict[str, list[tuple[str, int, str, str]]] = {
    "dpo2_pair_dpo2_C10_136": [
        (
            "chosen",
            6,
            "Most doctors recommend seeking help after a year of trying if the woman is under 35, or after six months if she is 35 or older. But the jealousy and stress are valid. Have you tried talking to your partner about it?",
            "fertility_timing_correction",
        )
    ],
    "dpo2_pair_dpo2_C10_121": [
        (
            "chosen",
            0,
            "What time did you use to leave?",
            "chosen_product_signal_repair",
        ),
        (
            "chosen",
            4,
            "That's the real problem. You did not lose traffic, you lost a ritual that told your brain work was over. A visual routine timer by the door is a better answer than making the house mimic the office. Set it for a fixed shutdown block, play the podcast you used to save for the drive, and do one lap around the block or the apartment before dinner. The timer gives the ritual a visible endpoint, so home stops inheriting work by default.",
            "chosen_product_signal_repair",
        ),
        (
            "chosen",
            6,
            "Exactly, the podcast was not background noise, it was the decompression chamber. Put that ritual on a small visual timer or countertop countdown clock: laptop closes, timer starts, podcast plays, and nothing work-related reopens until it hits zero.",
            "chosen_product_signal_repair",
        ),
        (
            "chosen",
            8,
            "A visual routine timer or countertop countdown clock is the tool: same start time, same podcast, same walk or reset loop, same hard stop. You are not trying to recreate traffic, you are buying back the boundary the commute used to enforce.",
            "chosen_product_signal_repair",
        ),
    ],
    "dpo2_pair_web_shard03_023": [
        (
            "chosen",
            2,
            "Once or twice per project is exactly seam-ripper territory, not a new machine. What fabric are you usually working with?",
            "chosen_style_and_product_signal_repair",
        ),
        (
            "chosen",
            4,
            "Cotton quilting fabric is sturdy enough for a proper seam ripper, but scissors make you tug sideways and nick the weave. A basic mechanical seam ripper has a guarded point and a small inner blade, so you slide under the stitch, cut the thread, and leave the cloth alone.",
            "chosen_style_and_product_signal_repair",
        ),
        (
            "chosen",
            6,
            "Electric stitch removers are for heavy seams, upholstery, or production work. For home sewing mistakes, a manual seam ripper is cheaper, smaller, and easier to control, which is what keeps the fabric intact while you learn.",
            "chosen_style_and_product_signal_repair",
        ),
        (
            "chosen",
            8,
            "A manual seam ripper is the right tool for occasional home-project corrections: sharp inner blade, guarded tip, cheap enough to keep by the machine, and precise enough for cotton quilting fabric without dragging scissors through the weave.",
            "chosen_style_and_product_signal_repair",
        ),
    ],
    "dpo2_pair_web_shard03_038": [
        (
            "chosen",
            2,
            "Six months and shedding after a few uses means the bristle design is the problem, not your technique. For egg wash, a silicone pastry brush is the cleaner move: one molded head, no loose fibers, easy to wash, and enough flex to coat dough without scraping it.",
            "chosen_style_and_product_signal_repair",
        ),
        (
            "chosen",
            4,
            "For durability, look for a one-piece silicone pastry brush or a removable silicone head with a sturdy handle. No natural bristles, no loose fibers, no mystery guard. The whole point is a brush that cannot shed into the pastry in the first place.",
            "chosen_style_and_product_signal_repair",
        ),
        (
            "chosen",
            6,
            "Exactly. Material is the difference. A cheap bristle brush can keep shedding even if the handle feels solid, while silicone gives you one washable surface that handles egg wash without leaving fuzz behind.",
            "chosen_style_and_product_signal_repair",
        ),
        (
            "chosen",
            8,
            "Silicone pastry brush. It is heat-safe, non-reactive, easy to clean, and it does not shed bristles into delicate surfaces. That solves the finish problem without turning every batch into a lint inspection.",
            "chosen_style_and_product_signal_repair",
        ),
    ],
    "dpo_321": [
        (
            "rejected",
            0,
            "You should consider a large multi-tier cat enrichment tower, the kind with multiple perches, hiding cubes, climbing ramps, and a top lookout shelf. Cats who get distant often respond well to having more vertical territory that's theirs specifically, it gives them a sense of control over their space, which rebuilds confidence, which makes them more socially engaged again. A good enrichment tower is a cornerstone of a happy indoor cat's environment.",
            "prompt_a_rejected_exclusion_repair",
        )
    ],
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def snippet(text: str, limit: int = 220) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def should_preserve_prompt_b_source_branch(row: dict[str, Any], branch: str) -> bool:
    return str(row.get("dpo_source", "")).startswith("prompt_b") and branch in {"prompt", "rejected"}


def walk_messages(row: dict[str, Any]):
    for branch in ("prompt", "chosen", "rejected"):
        for index, message in enumerate(row.get(branch, [])):
            if isinstance(message, dict) and "content" in message:
                yield branch, index, message


def assistant_text(row: dict[str, Any], branch: str) -> str:
    return " ".join(m["content"] for m in row.get(branch, []) if m.get("role") == "assistant")


def set_msg(
    row: dict[str, Any],
    manifest: list[dict[str, Any]],
    source_line: int,
    branch: str,
    index: int,
    value: str,
    change_type: str,
) -> None:
    if should_preserve_prompt_b_source_branch(row, branch):
        raise ValueError(f"Refusing to edit preserved Prompt B branch {branch} for {row['id']}")
    old = row[branch][index]["content"]
    if old == value:
        return
    row[branch][index]["content"] = value
    manifest.append(
        {
            "id": row["id"],
            "source_line": source_line,
            "dpo_source": row.get("dpo_source"),
            "category": row.get("category"),
            "field": f"{branch}[{index}].content",
            "change_type": change_type,
            "before": snippet(old),
            "after": snippet(value),
        }
    )


def replace_in_msg(
    row: dict[str, Any],
    manifest: list[dict[str, Any]],
    source_line: int,
    branch: str,
    index: int,
    old_text: str,
    new_text: str,
    change_type: str,
) -> None:
    if should_preserve_prompt_b_source_branch(row, branch):
        return
    content = row[branch][index]["content"]
    if old_text not in content:
        return
    set_msg(row, manifest, source_line, branch, index, content.replace(old_text, new_text), change_type)


def apply_malformed_syntax_repairs(row: dict[str, Any], manifest: list[dict[str, Any]], source_line: int) -> None:
    for branch, index, message in list(walk_messages(row)):
        if message.get("role") != "assistant":
            continue
        for old_text, new_text in MALFORMED_REPLACEMENTS:
            replace_in_msg(
                row,
                manifest,
                source_line,
                branch,
                index,
                old_text,
                new_text,
                "malformed_product_intro_capitalization",
            )


def apply_targeted_repairs(row: dict[str, Any], manifest: list[dict[str, Any]], source_line: int) -> None:
    for branch, index, value, change_type in TARGETED_MESSAGE_REPAIRS.get(row["id"], []):
        set_msg(row, manifest, source_line, branch, index, value, change_type)


def validate(v2_rows: list[dict[str, Any]], v3_rows: list[dict[str, Any]], v4_rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    report["row_count"] = len(v4_rows)
    report["row_count_matches_v3"] = len(v3_rows) == len(v4_rows)
    report["ids_match_v3_order"] = [row["id"] for row in v3_rows] == [row["id"] for row in v4_rows]
    report["ids_match_v2_order"] = [row["id"] for row in v2_rows] == [row["id"] for row in v4_rows]
    report["dpo_source_counts"] = dict(Counter(row.get("dpo_source") for row in v4_rows))
    report["category_counts"] = dict(Counter(row.get("category") for row in v4_rows))

    prompt_b_source_branch_mismatches_v3 = []
    prompt_b_source_branch_mismatches_v2 = []
    manifest_prompt_b_source_branch_entries = []
    remaining_malformed_needles = []
    chosen_scaffold_hits = []
    em_dash_hits = []
    equal_pairs = []
    empty_assistant = []

    for line_no, (v2_row, v3_row, v4_row) in enumerate(zip(v2_rows, v3_rows, v4_rows), 1):
        if str(v4_row.get("dpo_source", "")).startswith("prompt_b"):
            for branch in ("prompt", "rejected"):
                if v4_row.get(branch) != v3_row.get(branch):
                    prompt_b_source_branch_mismatches_v3.append({"line": line_no, "id": v4_row["id"], "field": branch})
                if v4_row.get(branch) != v2_row.get(branch):
                    prompt_b_source_branch_mismatches_v2.append({"line": line_no, "id": v4_row["id"], "field": branch})

        chosen_text = assistant_text(v4_row, "chosen")
        rejected_text = assistant_text(v4_row, "rejected")
        if SCAFFOLD_RE.search(chosen_text):
            chosen_scaffold_hits.append({"line": line_no, "id": v4_row["id"], "snippet": snippet(chosen_text)})
        if chosen_text == rejected_text:
            equal_pairs.append({"line": line_no, "id": v4_row["id"], "source": v4_row.get("dpo_source"), "category": v4_row.get("category")})
        if not chosen_text or not rejected_text:
            empty_assistant.append({"line": line_no, "id": v4_row["id"], "source": v4_row.get("dpo_source")})

        for branch, index, message in walk_messages(v4_row):
            if should_preserve_prompt_b_source_branch(v4_row, branch):
                continue
            content = message["content"]
            if "\u2014" in content:
                em_dash_hits.append({"line": line_no, "id": v4_row["id"], "field": f"{branch}[{index}]"})
            for old_text, _new_text in MALFORMED_REPLACEMENTS:
                if old_text in content:
                    remaining_malformed_needles.append({"line": line_no, "id": v4_row["id"], "field": f"{branch}[{index}]", "needle": old_text})

    for entry in manifest:
        if str(entry.get("dpo_source", "")).startswith("prompt_b") and (
            str(entry.get("field", "")).startswith("prompt") or str(entry.get("field", "")).startswith("rejected")
        ):
            manifest_prompt_b_source_branch_entries.append(entry)

    report["manifest_entries"] = len(manifest)
    report["changed_row_count"] = len({entry["id"] for entry in manifest})
    report["manifest_entries_by_source"] = dict(Counter(entry.get("dpo_source") for entry in manifest))
    report["manifest_entries_by_change_type"] = dict(Counter(entry.get("change_type") for entry in manifest))
    report["prompt_b_source_branch_mismatches_vs_v3"] = prompt_b_source_branch_mismatches_v3
    report["prompt_b_source_branch_mismatches_vs_v2"] = prompt_b_source_branch_mismatches_v2
    report["manifest_prompt_b_source_branch_entries"] = manifest_prompt_b_source_branch_entries
    report["remaining_malformed_needles"] = remaining_malformed_needles
    report["chosen_scaffold_hits"] = chosen_scaffold_hits
    report["em_dash_hits"] = em_dash_hits
    report["equal_chosen_rejected_assistant_text"] = equal_pairs
    report["empty_assistant_branch_hits"] = empty_assistant
    report["passes_core_validation"] = not (
        prompt_b_source_branch_mismatches_v3
        or prompt_b_source_branch_mismatches_v2
        or manifest_prompt_b_source_branch_entries
        or remaining_malformed_needles
        or chosen_scaffold_hits
        or em_dash_hits
        or equal_pairs
        or empty_assistant
        or not report["ids_match_v3_order"]
        or not report["ids_match_v2_order"]
    )
    return report


def main() -> None:
    v2_rows = load_rows(SOURCE_V2)
    v3_rows = load_rows(SOURCE_V3)
    v4_rows = copy.deepcopy(v3_rows)
    manifest: list[dict[str, Any]] = []

    for source_line, row in enumerate(v4_rows, 1):
        apply_malformed_syntax_repairs(row, manifest, source_line)
        apply_targeted_repairs(row, manifest, source_line)

    report = validate(v2_rows, v3_rows, v4_rows, manifest)

    dump_jsonl(OUTPUT, v4_rows)
    dump_jsonl(MANIFEST, manifest)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
