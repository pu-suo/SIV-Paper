"""
Vampire Theorem Prover Interface.

Ported and extended from SIV_Evaluation_Framework (3).ipynb, Cells 5, 12.

Vampire is optional: if it cannot be located the interface degrades gracefully,
returning None ("unresolved") rather than raising.  All code that calls
check_entailment() must handle the None case.

Usage:
    from siv.vampire_interface import check_entailment, setup_vampire

    # Returns True (proved) / False (disproved) / None (timeout or unavailable)
    result = check_entailment("all x.(Dog(x) -> Animal(x))", "all x.(Dog(x) -> Animal(x))")
"""
import io
import os
import subprocess
import platform
import stat
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from siv.fol_utils import parse_fol, convert_to_tptp, NLTK_AVAILABLE

# ── Vampire availability ───────────────────────────────────────────────────────

_VAMPIRE_PATH: Optional[str] = None   # Resolved at first use

# Download URLs per platform+arch (Vampire 5.0.1 zip releases)
_VAMPIRE_VERSION = "v5.0.1"
_VAMPIRE_BASE = f"https://github.com/vprover/vampire/releases/download/{_VAMPIRE_VERSION}"
_VAMPIRE_URLS = {
    ("Linux",  "X64"):   f"{_VAMPIRE_BASE}/vampire-Linux-X64.zip",
    ("Linux",  "ARM64"): f"{_VAMPIRE_BASE}/vampire-Linux-ARM64.zip",
    ("Darwin", "X64"):   f"{_VAMPIRE_BASE}/vampire-macOS-X64.zip",
    ("Darwin", "ARM64"): f"{_VAMPIRE_BASE}/vampire-macOS-ARM64.zip",
}


def _detect_arch() -> str:
    machine = platform.machine().lower()
    return "ARM64" if ("arm" in machine or "aarch" in machine) else "X64"


def _find_vampire() -> Optional[str]:
    """Search for the Vampire binary in common locations."""
    candidates = [
        os.environ.get("VAMPIRE_PATH", ""),
        "./vampire",
        "/usr/local/bin/vampire",
        "/usr/bin/vampire",
        str(Path.home() / "vampire"),
        str(Path(__file__).parent.parent / "vampire"),
    ]
    for path in candidates:
        if path and Path(path).is_file() and os.access(path, os.X_OK):
            return path
    return None


def setup_vampire(target_dir: str = ".") -> Optional[str]:
    """
    Download and install the Vampire binary for the current platform.

    Downloads the appropriate zip from the Vampire GitHub releases, extracts
    all contents to a temporary directory, walks the tree to find the Vampire
    binary (however it is nested), copies it to *target_dir*/vampire, and
    makes it executable.  Returns the path to the installed binary, or None
    if installation failed.  Call this once in a notebook setup cell.
    """
    import shutil
    import tempfile

    system = platform.system()
    arch = _detect_arch()
    url = _VAMPIRE_URLS.get((system, arch))
    if url is None:
        print(f"[Vampire] Unsupported platform/arch: {system}/{arch}. Install Vampire manually.")
        return None

    dest = Path(target_dir) / "vampire"
    if dest.exists() and os.access(str(dest), os.X_OK):
        print(f"[Vampire] Already installed at {dest}")
        return str(dest)

    # Ensure the target directory exists before writing into it
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Vampire] Downloading {_VAMPIRE_VERSION} ({system}/{arch}) from {url} …")
    try:
        with urllib.request.urlopen(url) as response:
            zip_data = response.read()

        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                zf.extractall(tmp)

            # Walk the extracted tree to find the Vampire binary
            found: Optional[str] = None
            for root, _, files in os.walk(tmp):
                for name in files:
                    if "vampire" in name.lower() and not name.endswith(".zip"):
                        found = os.path.join(root, name)
                        break
                if found:
                    break

            if found is None:
                print("[Vampire] Binary not found inside zip.")
                return None

            shutil.copy2(found, str(dest))

        dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"[Vampire] Installed at {dest}")
        return str(dest)
    except Exception as e:
        print(f"[Vampire] Download failed: {e}")
        return None


def is_vampire_available() -> bool:
    """Return True if Vampire is available and executable."""
    global _VAMPIRE_PATH
    if _VAMPIRE_PATH is None:
        _VAMPIRE_PATH = _find_vampire()
    return _VAMPIRE_PATH is not None


# ── TPTP file generation ──────────────────────────────────────────────────────

def _build_tptp(premises_fol: List[str], conclusion_fol: str) -> Optional[str]:
    """
    Build a TPTP problem string from FOL premise strings and a conclusion.

    Returns None if any string fails to parse.
    """
    if not NLTK_AVAILABLE:
        return None

    premise_exprs = [parse_fol(p) for p in premises_fol]
    conc_expr = parse_fol(conclusion_fol)

    if not all(p is not None for p in premise_exprs) or conc_expr is None:
        return None

    lines = []
    for i, expr in enumerate(premise_exprs):
        lines.append(f"fof(premise{i}, axiom, {convert_to_tptp(expr)}).")
    lines.append(f"fof(goal, conjecture, {convert_to_tptp(conc_expr)}).")
    return "\n".join(lines)


# ── Core prover call ──────────────────────────────────────────────────────────

def _run_vampire(tptp_input: str, timeout: int) -> Tuple[Optional[bool], str]:
    """
    Run Vampire on *tptp_input* and return (proved, raw_output).

    proved:
      True   → Vampire found a refutation (conjecture proved)
      False  → Vampire found a counter-model (conjecture refuted)
      None   → timeout, error, or inconclusive
    """
    global _VAMPIRE_PATH
    if _VAMPIRE_PATH is None:
        _VAMPIRE_PATH = _find_vampire()
    if _VAMPIRE_PATH is None:
        return None, "Vampire not available"

    try:
        process = subprocess.Popen(
            [_VAMPIRE_PATH, "--input_syntax", "tptp", "-t", str(timeout)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(tptp_input, timeout=timeout + 5)
        output = stdout + stderr

        if "Termination reason: Refutation" in output:
            return True, output
        if "Termination reason: CounterSatisfiable" in output:
            return False, output
        # Satisfiable (when goal is $false): the axiom set has a model, so
        # it is *not* unsatisfiable.
        if "Termination reason: Satisfiable" in output:
            return False, output
        return None, output

    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except Exception:
            pass
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


# ── Public API ────────────────────────────────────────────────────────────────

def check_entailment(
    premise_fol: str,
    conclusion_fol: str,
    timeout: int = 5,
) -> Optional[bool]:
    """
    Check whether *premise_fol* entails *conclusion_fol*.

    Returns:
      True   — entailment confirmed by Vampire
      False  — Vampire found a counter-model (non-entailment)
      None   — Vampire unavailable, timeout, or parse failure
    """
    tptp = _build_tptp([premise_fol], conclusion_fol)
    if tptp is None:
        return None
    proved, _ = _run_vampire(tptp, timeout)
    return proved


def check_entailment_multi(
    premises_fol: List[str],
    conclusion_fol: str,
    timeout: int = 5,
) -> Optional[bool]:
    """
    Check whether the conjunction of *premises_fol* entails *conclusion_fol*.

    Returns True / False / None as per check_entailment().
    """
    tptp = _build_tptp(premises_fol, conclusion_fol)
    if tptp is None:
        return None
    proved, _ = _run_vampire(tptp, timeout)
    return proved


VampireVerdict = str  # Literal["sat", "unsat", "timeout", "unknown"]


def vampire_check(
    fol_a: str,
    fol_b: str,
    check: str,  # Literal["unsat", "entails"]
    timeout: int = 5,
    axioms: Optional[List[str]] = None,
) -> str:
    """Vampire-check a pair of FOL strings. Returns one of:
    ``"unsat"``, ``"sat"``, ``"timeout"``, ``"unknown"``.

    - ``check="entails"``: does ``(fol_a ∧ ⋀axioms)`` entail ``fol_b``?
      Verdict is the status of ``(fol_a ∧ ⋀axioms ∧ ¬fol_b)``. ``"unsat"``
      ⇒ entailment proved.
    - ``check="unsat"``: is ``(fol_a ∧ fol_b ∧ ⋀axioms)`` unsatisfiable?
      ``"unsat"`` ⇒ the formulas are jointly inconsistent under the
      supplied axioms.

    ``axioms`` is an optional list of FOL-string side-axioms added to
    Vampire's axiom set before the check. Used by the witness-axiom
    protocol; omit to preserve free-domain semantics.
    """
    if check not in ("unsat", "entails"):
        raise ValueError(f"unknown check mode: {check!r}")

    extra = list(axioms) if axioms else []
    if check == "entails":
        tptp = _build_tptp([fol_a, *extra], fol_b)
    else:
        tptp = _build_unsat_tptp([fol_a, fol_b, *extra])

    if tptp is None:
        return "unknown"
    proved, output = _run_vampire(tptp, timeout)
    if proved is True:
        return "unsat"
    if proved is False:
        return "sat"
    if output == "Timeout":
        return "timeout"
    return "unknown"


def _build_unsat_tptp(fol_axioms: List[str]) -> Optional[str]:
    """Build a TPTP problem whose conjecture is ``$false``.

    Vampire refutes iff the axioms are jointly inconsistent — i.e., the
    conjunction of the axioms is unsatisfiable.
    """
    if not NLTK_AVAILABLE:
        return None
    exprs = [parse_fol(p) for p in fol_axioms]
    if any(e is None for e in exprs):
        return None
    lines = []
    for i, expr in enumerate(exprs):
        lines.append(f"fof(axiom{i}, axiom, {convert_to_tptp(expr)}).")
    lines.append("fof(goal, conjecture, $false).")
    return "\n".join(lines)


def prove_strict(
    premises_str: List[str],
    conclusion_str: str,
    timeout: int = 5,
) -> Tuple[str, Optional[str]]:
    """
    Full NLI-style check: entailment / contradiction / neutral.

    Returns (label, proof_info) where label ∈ {"entailment", "contradiction", "neutral"}.
    Compatible with the original notebook interface.
    """
    tptp_pos = _build_tptp(premises_str, conclusion_str)
    if tptp_pos is None:
        return "neutral", "Invalid FOL syntax"

    proved, proof = _run_vampire(tptp_pos, timeout)
    if proved is True:
        return "entailment", proof

    # Try negation
    neg_conc = f"-({conclusion_str})"
    tptp_neg = _build_tptp(premises_str, neg_conc)
    if tptp_neg is not None:
        neg_proved, neg_proof = _run_vampire(tptp_neg, timeout)
        if neg_proved is True:
            return "contradiction", neg_proof

    return "neutral", proof or "No proof found"
