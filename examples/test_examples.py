#!/usr/bin/env python
"""
Test script to verify all training example variants work correctly.

This script runs small training runs with different configurations to ensure
the training pipeline works for all combinations of:
- MNIST and regression datasets
- SimpleCell and AOCCell models
- Different num-layers (1, 2)
- Different fixed-point-init (zeros, x_proj, random)

Usage:
    python examples/test_examples.py
    python examples/test_examples.py --verbose
    python examples/test_examples.py --quick  # Even faster tests
"""
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestCase:
    name: str
    script: str
    args: List[str]
    expected_success: bool = True


def run_test(test: TestCase, verbose: bool = False) -> bool:
    """Run a single test case and return success status."""
    cmd = [sys.executable, test.script] + test.args
    
    print(f"  Running: {test.name}...", end=" ", flush=True)
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        elapsed = time.time() - start_time
        
        success = (result.returncode == 0) == test.expected_success
        
        if success:
            print(f"✓ ({elapsed:.1f}s)")
        else:
            print(f"✗ ({elapsed:.1f}s)")
            
        if verbose or not success:
            if result.stdout:
                print(f"    STDOUT:\n{indent(result.stdout, 6)}")
            if result.stderr:
                print(f"    STDERR:\n{indent(result.stderr, 6)}")
                
        return success
        
    except subprocess.TimeoutExpired:
        print("✗ (TIMEOUT)")
        return False
    except Exception as e:
        print(f"✗ (ERROR: {e})")
        return False


def indent(text: str, spaces: int) -> str:
    """Indent multi-line text."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.strip().split("\n")[-20:])  # Last 20 lines


def get_mnist_tests(quick: bool = False) -> List[TestCase]:
    """Get MNIST test cases."""
    epochs = "1" if quick else "2"
    base_args = ["--epochs", epochs, "--hidden-size", "16", "--batch-size", "128"]
    # AOCCell needs hidden_size != 16 for 1-layer to avoid 16-variable mode (incomplete config)
    aoc_base_args = ["--epochs", epochs, "--hidden-size", "24", "--batch-size", "128"]
    
    tests = [
        # SimpleCell tests
        TestCase(
            name="MNIST SimpleCell 1-layer zeros",
            script="examples/train_mnist.py",
            args=base_args + ["--num-layers", "1", "--fixed-point-init", "zeros"],
        ),
        TestCase(
            name="MNIST SimpleCell 1-layer x_proj",
            script="examples/train_mnist.py",
            args=base_args + ["--num-layers", "1", "--fixed-point-init", "x_proj"],
        ),
        TestCase(
            name="MNIST SimpleCell 2-layer zeros",
            script="examples/train_mnist.py",
            args=base_args + ["--num-layers", "2", "--fixed-point-init", "zeros"],
        ),
        TestCase(
            name="MNIST SimpleCell 2-layer random",
            script="examples/train_mnist.py",
            args=base_args + ["--num-layers", "2", "--fixed-point-init", "random"],
        ),
        # AOCCell tests (use hidden_size=24 to avoid 16-var mode)
        TestCase(
            name="MNIST AOCCell 1-layer zeros",
            script="examples/train_mnist.py",
            args=aoc_base_args + ["--num-layers", "1", "--fixed-point-init", "zeros", "--use-aoc-cell"],
        ),
        TestCase(
            name="MNIST AOCCell 1-layer x_proj",
            script="examples/train_mnist.py",
            args=aoc_base_args + ["--num-layers", "1", "--fixed-point-init", "x_proj", "--use-aoc-cell"],
        ),
        TestCase(
            name="MNIST AOCCell 2-layer zeros",
            script="examples/train_mnist.py",
            args=aoc_base_args + ["--num-layers", "2", "--fixed-point-init", "zeros", "--use-aoc-cell"],
        ),
    ]
    
    return tests


def get_regression_tests(quick: bool = False) -> List[TestCase]:
    """Get regression test cases."""
    epochs = "5" if quick else "10"
    base_args = ["--epochs", epochs, "--hidden-size", "16", "--n-points", "100", "--batch-size", "32"]
    # AOCCell needs hidden_size != 16 for 1-layer to avoid 16-variable mode (incomplete config)
    aoc_base_args = ["--epochs", epochs, "--hidden-size", "24", "--n-points", "100", "--batch-size", "32"]
    
    tests = [
        # SimpleCell tests with different datasets
        TestCase(
            name="Regression sinusoidal SimpleCell 1-layer",
            script="examples/train_regression.py",
            args=base_args + ["--dataset", "sinusoidal", "--num-layers", "1", "--fixed-point-init", "zeros"],
        ),
        TestCase(
            name="Regression polynomial SimpleCell 2-layer",
            script="examples/train_regression.py",
            args=base_args + ["--dataset", "polynomial", "--num-layers", "2", "--fixed-point-init", "x_proj"],
        ),
        TestCase(
            name="Regression gaussian SimpleCell 1-layer random",
            script="examples/train_regression.py",
            args=base_args + ["--dataset", "gaussian", "--num-layers", "1", "--fixed-point-init", "random"],
        ),
        # AOCCell tests (use hidden_size=24 to avoid 16-var mode)
        TestCase(
            name="Regression sinusoidal AOCCell 1-layer",
            script="examples/train_regression.py",
            args=aoc_base_args + ["--dataset", "sinusoidal", "--num-layers", "1", "--fixed-point-init", "zeros", "--use-aoc-cell"],
        ),
        TestCase(
            name="Regression polynomial AOCCell 2-layer",
            script="examples/train_regression.py",
            args=aoc_base_args + ["--dataset", "polynomial", "--num-layers", "2", "--fixed-point-init", "x_proj", "--use-aoc-cell"],
        ),
    ]
    
    return tests


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test all training example variants")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--quick", "-q", action="store_true", help="Run even faster tests")
    parser.add_argument("--mnist-only", action="store_true", help="Only run MNIST tests")
    parser.add_argument("--regression-only", action="store_true", help="Only run regression tests")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing AOC Training Examples")
    print("=" * 60)
    
    all_tests = []
    
    if not args.regression_only:
        print("\n[MNIST Tests]")
        mnist_tests = get_mnist_tests(quick=args.quick)
        all_tests.extend(mnist_tests)
        
    if not args.mnist_only:
        print("\n[Regression Tests]")
        regression_tests = get_regression_tests(quick=args.quick)
        all_tests.extend(regression_tests)
    
    # Run all tests
    results = []
    start_time = time.time()
    
    for test in all_tests:
        if test.script.endswith("train_mnist.py") and args.regression_only:
            continue
        if test.script.endswith("train_regression.py") and args.mnist_only:
            continue
            
        # Print section header when switching test types
        if results and all_tests[len(results)].script != all_tests[len(results)-1].script:
            print()
            
        success = run_test(test, verbose=args.verbose)
        results.append((test.name, success))
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(results)} passed, {failed} failed")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)
    
    if failed > 0:
        print("\nFailed tests:")
        for name, success in results:
            if not success:
                print(f"  - {name}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
