from pathlib import Path
import re

TARGET = Path("fractal_dtes_aco_zeta_crosschannel.py")

if not TARGET.exists():
    raise SystemExit(f"File not found: {TARGET}")

text = TARGET.read_text(encoding="utf-8")

pattern = re.compile(
    r"(agent_score\s*=\s*self\.compute_agent_value\(ant,\s*current,\s*nxt\)\s*\n)",
    re.MULTILINE
)

replacement = r"""\1
            # --- DTES exploration boost ---
            visit_bonus = 1.0 / (1.0 + self.nodes[nxt].visit_count) ** 0.5
            level_bonus = 0.15 * (self.nodes[nxt].level / max(1, self.cfg.tree_depth))
            agent_score += 0.75 * visit_bonus + level_bonus
"""

new_text, count = pattern.subn(replacement, text)

if count == 0:
    raise SystemExit("Pattern not found. Check file structure.")

TARGET.write_text(new_text, encoding="utf-8")

print(f"[OK] Patched exploration boost ({count} locations)")