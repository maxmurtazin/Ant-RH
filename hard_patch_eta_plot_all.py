#!/usr/bin/env python3
from pathlib import Path
import re

TARGET = Path("fractal_dtes_aco_zeta_eta.py")

NEW_METHOD = """
    def plot_all(self, candidates, out_dir):
        from pathlib import Path
        import json

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            t_grid = getattr(self, "t_grid", None)
            if t_grid is None:
                t_grid = getattr(self, "t", None)
            if t_grid is None:
                t_grid = getattr(self, "grid", None)

            energies = getattr(self, "energies", None)
            if energies is None:
                energies = getattr(self, "energy", None)
            if energies is None:
                energies = getattr(self, "log_abs", None)
            if energies is None:
                energies = getattr(self, "log_abs_zeta", None)

            if t_grid is not None and energies is not None:
                fig_path = out_dir / "energy_landscape.png"
                plt.figure(figsize=(10, 4))
                plt.plot(t_grid, energies, linewidth=1)

                xs, ys = [], []
                for c in candidates or []:
                    if isinstance(c, dict):
                        x = c.get("t_mid", c.get("t", c.get("center", None)))
                        y = c.get("energy", c.get("score", None))
                    else:
                        x = getattr(c, "t_mid", getattr(c, "t", getattr(c, "center", None)))
                        y = getattr(c, "energy", getattr(c, "score", None))

                    if x is not None:
                        xs.append(x)
                        if y is None:
                            try:
                                y = float(np.nanmin(energies))
                            except Exception:
                                y = 0.0
                        ys.append(y)

                if xs:
                    plt.scatter(xs, ys, marker="x")

                plt.xlabel("t")
                plt.ylabel("energy / log|zeta|")
                plt.title("DTES-Zeta Energy Landscape")
                plt.tight_layout()
                plt.savefig(fig_path, dpi=180)
                plt.close()
                paths["energy_landscape"] = str(fig_path)

            history = getattr(self, "history", None)
            if history is None:
                history = getattr(self, "aco_history", None)

            if history:
                fig_path = out_dir / "aco_convergence.png"
                xs = list(range(1, len(history) + 1))
                best, eta = [], []

                for h in history:
                    if isinstance(h, dict):
                        best.append(h.get("best_energy", h.get("best_E", h.get("energy", None))))
                        eta.append(h.get("eta_s", h.get("eta", None)))
                    else:
                        best.append(getattr(h, "best_energy", None))
                        eta.append(getattr(h, "eta_s", None))

                plt.figure(figsize=(10, 4))
                if any(v is not None for v in best):
                    plt.plot(xs, [v if v is not None else float("nan") for v in best], label="best_energy")
                if any(v is not None for v in eta):
                    plt.plot(xs, [v if v is not None else float("nan") for v in eta], label="eta_s")
                plt.xlabel("iteration")
                plt.title("ACO / ETA History")
                plt.legend()
                plt.tight_layout()
                plt.savefig(fig_path, dpi=180)
                plt.close()
                paths["aco_convergence"] = str(fig_path)

            cand_path = out_dir / "candidates.json"
            with open(cand_path, "w", encoding="utf-8") as f:
                json.dump(candidates, f, indent=2, default=str)
            paths["candidates"] = str(cand_path)

        except Exception as exc:
            error_path = out_dir / "plot_error.txt"
            error_path.write_text(str(exc), encoding="utf-8")
            paths["plot_error"] = str(error_path)

        return paths

"""


def find_method_bounds(text: str, method_name: str = "plot_all") -> tuple[int, int]:
    m = re.search(rf"^    def\s+{re.escape(method_name)}\s*\(", text, flags=re.M)
    if not m:
        raise RuntimeError(f"Could not find method: def {method_name}(...)")
    start = m.start()
    rest = text[m.end():]
    next_m = re.search(r"^(    def\s+\w+\s*\(|class\s+\w+|def\s+\w+)", rest, flags=re.M)
    end = len(text) if not next_m else m.end() + next_m.start()
    return start, end


def main():
    if not TARGET.exists():
        raise SystemExit(f"File not found: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(TARGET.suffix + ".bak_hard_plotfix")
    backup.write_text(text, encoding="utf-8")

    start, end = find_method_bounds(text, "plot_all")
    text = text[:start] + NEW_METHOD + text[end:]

    compile(text, str(TARGET), "exec")
    TARGET.write_text(text, encoding="utf-8")

    print("[OK] hard patched plot_all")
    print(f"[OK] backup: {backup}")


if __name__ == "__main__":
    main()
