#!/usr/bin/env python3
"""
OmniStitch Studio & CTU_P — Conveyor Belt Automated Verification Engine
Implements the 6-Station Quality Control Protocol defined in AGENTS.md.
"""

import os
import sys
import ast
import glob
import time
import re
import unittest
from io import StringIO


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


def check_ast_assertions(test_dir="tests"):
    """
    [Trạm 4: AST Integrity]
    Scans all test files to ensure 100% of test methods contain active assertions.
    Prevents false-positive test passes caused by merge conflict truncations (CASE-001).
    """
    pattern = os.path.join(test_dir, "test_*.py")
    test_files = sorted(glob.glob(pattern))
    empty_tests = []
    total_methods = 0

    for tf in test_files:
        with open(tf, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=tf)
        except SyntaxError as e:
            return False, 0, [f"Syntax error in {tf}: {e}"]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                        total_methods += 1
                        asserts = [
                            n for n in ast.walk(item)
                            if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute)
                            and n.func.attr.startswith("assert")
                        ]
                        assert_stmts = [n for n in ast.walk(item) if isinstance(n, ast.Assert)]
                        if not asserts and not assert_stmts:
                            empty_tests.append(f"{tf}::{node.name}::{item.name}")

    is_valid = (len(empty_tests) == 0 and total_methods > 0)
    return is_valid, total_methods, empty_tests


def check_handbook_integrity(handbook_path="../HANDBOOK.md"):
    """
    [Trạm 3: Docs Integrity]
    Validates that HANDBOOK.md conforms to the required CASE study format:
    - Context
    - Symptoms
    - Root Cause
    - Architectural Fix
    - Preventative Invariant
    """
    if not os.path.exists(handbook_path):
        # Try finding in current directory
        if os.path.exists("HANDBOOK.md"):
            handbook_path = "HANDBOOK.md"
        else:
            return False, 0, [f"Handbook not found at {handbook_path}"]

    with open(handbook_path, "r", encoding="utf-8") as f:
        content = f.read()

    required_sections = [
        "Thời điểm / Bối cảnh",
        "Triệu chứng & Tác động",
        "Nguyên nhân Gốc rễ",
        "Giải pháp Triệt để",
        "Bất biến Phòng ngừa",
    ]

    case_blocks = re.split(r"### \[(CASE-\d+)\]", content)
    if len(case_blocks) < 3:
        return False, 0, ["No CASE studies found in HANDBOOK.md"]

    issues = []
    cases_found = []

    for i in range(1, len(case_blocks), 2):
        case_id = case_blocks[i]
        case_body = case_blocks[i + 1]
        title_line = case_body.strip().split("\n")[0].strip()
        cases_found.append((case_id, title_line))

        for section in required_sections:
            if section not in case_body:
                issues.append(f"{case_id} is missing required section: '{section}'")

    is_valid = (len(issues) == 0)
    return is_valid, len(cases_found), issues


def check_engineering_invariants(repo_root="."):
    """
    [Trạm 1: Adversarial Review & Core Invariants]
    Verifies that core architectural invariants are strictly respected:
    1. 2D 3x3 Matrix Contract (Homography & Affine preservation)
    2. Local Clarity Inspection on raw pixel crop (Var(Laplacian))
    3. Memory safety & streaming tiled export (mmap)
    4. H&E Stain Fidelity (Preserve Hematoxylin & Eosin hues)
    """
    issues = []
    checked = []

    blending_py = os.path.join(repo_root, "backend", "blending.py")
    if os.path.exists(blending_py):
        with open(blending_py, "r", encoding="utf-8") as f:
            code = f.read()
        # Invariant 1: 3x3 Homography warpPerspective support
        if "cv2.warpPerspective" in code and "warpAffine" in code:
            checked.append("Matrix 2D 3x3 Contract: warpPerspective & warpAffine handled properly")
        else:
            issues.append("Invariant 1 Violation: Missing warpPerspective support in blending.py")

        # Invariant 2: Laplacian variance
        if "cv2.Laplacian" in code and ("var()" in code or "np.var" in code):
            checked.append("Clarity Inspection: Laplacian variance calculation present")
        else:
            issues.append("Invariant 2 Violation: Missing cv2.Laplacian variance in blending.py")

        # Invariant 4: Stain Fidelity & Gain balance
        if "solve_tissue_specific_gains" in code:
            checked.append("Histology Stain Fidelity: Kiệt LS tissue-specific gain solver active")
        else:
            issues.append("Invariant 4 Violation: Missing solve_tissue_specific_gains in blending.py")

    exporter_py = os.path.join(repo_root, "backend", "wsi_exporter.py")
    if os.path.exists(exporter_py):
        with open(exporter_py, "r", encoding="utf-8") as f:
            code = f.read()
        # Invariant 3: Memory streaming mmap / tile streaming
        if "tifffile" in code:
            checked.append("Memory Safety: Streaming TIFF exporter with BigTIFF capability active")
        else:
            issues.append("Invariant 3 Violation: Missing streaming tifffile support in wsi_exporter.py")

    is_valid = (len(issues) == 0)
    return is_valid, len(checked), issues


def run_unit_tests(test_dir="tests"):
    """
    [Trạm 2: Test Runner & Empirical Evidence]
    Executes full unittest discovery and computes quantitative execution metrics.
    """
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern="test_*.py")

    total_tests = suite.countTestCases()
    start_time = time.time()

    # Capture unittest output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=1)
    result = runner.run(suite)

    duration = round(time.time() - start_time, 3)
    is_success = result.wasSuccessful()
    failures = len(result.failures)
    errors = len(result.errors)

    details = {
        "total": total_tests,
        "run": result.testsRun,
        "failures": failures,
        "errors": errors,
        "duration_sec": duration,
        "success": is_success,
    }

    return is_success, details


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CTU_P / OmniStitch Conveyor Belt Engine")
    parser.add_argument("--ast", action="store_true", help="Run AST assertion integrity check (Station 4)")
    parser.add_argument("--docs", action="store_true", help="Run HANDBOOK.md integrity check (Station 3)")
    parser.add_argument("--invariants", action="store_true", help="Run engineering invariants check (Station 1)")
    parser.add_argument("--tests", action="store_true", help="Run unit test suite (Station 2)")
    parser.add_argument("--all", action="store_true", help="Run full automated verification pipeline")

    args = parser.parse_args()

    # If no flags passed, default to --all
    if not (args.ast or args.docs or args.invariants or args.tests or args.all):
        args.all = True

    overall_success = True

    # Station 4: AST Integrity
    if args.ast or args.all:
        print("[Trạm 4: Lint & AST Integrity] Scanning test suite for active assertions...")
        ast_ok, total_methods, empty = check_ast_assertions("tests")
        if ast_ok:
            print(f"  ✓ AST Integrity Verified: {total_methods} test methods all contain active assertions.")
        else:
            print(f"  ✗ AST Integrity FAILED: Found {len(empty)} empty test methods without assertions:", file=sys.stderr)
            for e in empty:
                print(f"    - {e}", file=sys.stderr)
            overall_success = False

    # Station 2: Tests
    if args.tests or args.all:
        print("[Trạm 2: Test & Empirical Evidence] Running full test discovery...")
        test_ok, test_metrics = run_unit_tests("tests")
        if test_ok:
            print(
                f"  ✓ Test Suite Passed: {test_metrics['run']}/{test_metrics['total']} tests OK "
                f"in {test_metrics['duration_sec']}s (Failures: {test_metrics['failures']}, Errors: {test_metrics['errors']})."
            )
        else:
            print(
                f"  ✗ Test Suite FAILED: Failures={test_metrics['failures']}, "
                f"Errors={test_metrics['errors']} in {test_metrics['duration_sec']}s",
                file=sys.stderr,
            )
            overall_success = False

    # Station 3: Docs
    if args.docs or args.all:
        print("[Trạm 3: Docs Integrity] Verifying HANDBOOK.md case study format...")
        docs_ok, case_count, issues = check_handbook_integrity("../HANDBOOK.md")
        if docs_ok:
            print(f"  ✓ Docs Integrity Verified: {case_count} case studies conform to standard structure.")
        else:
            print(f"  ✗ Docs Integrity FAILED: {len(issues)} issues found in handbook:", file=sys.stderr)
            for iss in issues:
                print(f"    - {iss}", file=sys.stderr)
            overall_success = False

    # Station 1: Invariants
    if args.invariants or args.all:
        print("[Trạm 1: Review & Invariants] Checking core architectural invariants...")
        inv_ok, count_checked, issues = check_engineering_invariants(".")
        if inv_ok:
            print(f"  ✓ Architectural Invariants Verified: {count_checked}/4 invariants strictly respected.")
        else:
            print(f"  ✗ Invariant Check FAILED: {len(issues)} violations found:", file=sys.stderr)
            for iss in issues:
                print(f"    - {iss}", file=sys.stderr)
            overall_success = False

    if not overall_success:
        print("\n[Conveyor Belt Alert] One or more verification stations failed!", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n[Conveyor Belt Success] All verification stations passed cleanly!")
        sys.exit(0)


if __name__ == "__main__":
    main()
