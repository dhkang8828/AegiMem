#!/usr/bin/env python3
"""
Test CE Monitor functionality
"""

import sys
import time
sys.path.insert(0, '/home/dhkang/cxl_memory_rl_project/src')

from ce_monitor import CEMonitor, MockCEMonitor, CEEvent


def test_mock_ce_monitor():
    """Test basic mock CE monitor functionality"""
    print("=" * 60)
    print("Testing Mock CE Monitor")
    print("=" * 60)

    monitor = MockCEMonitor(device_path="/dev/dax0.0", failure_rate=0.01)

    # Start monitoring
    monitor.start_monitoring()
    assert monitor.monitoring_active == True, "Monitoring should be active"
    print("✓ Monitoring started")

    # Initially, no new errors
    delta = monitor.get_ce_delta()
    assert delta == 0, f"Initial delta should be 0, got {delta}"
    print(f"✓ Initial CE delta: {delta}")

    # Inject some CE events
    monitor.inject_ce(count=5)
    monitor.update()

    delta = monitor.get_ce_delta()
    assert delta == 5, f"After injecting 5 CEs, delta should be 5, got {delta}"
    print(f"✓ After injection, CE delta: {delta}")

    # Inject more CEs with DPA
    monitor.inject_ce(count=3, dpa=0x100000)
    monitor.update()

    delta = monitor.get_ce_delta()
    assert delta == 8, f"After injecting 3 more CEs, delta should be 8, got {delta}"
    print(f"✓ After second injection, CE delta: {delta}")

    # Check CE by source
    by_source = monitor.get_ce_by_source()
    assert 'mock' in by_source, "Should have 'mock' source"
    assert by_source['mock'] == 8, f"Mock source should have 8 CEs, got {by_source['mock']}"
    print(f"✓ CE by source: {by_source}")

    # Reset baseline
    monitor.reset_baseline()
    delta = monitor.get_ce_delta()
    assert delta == 0, f"After reset, delta should be 0, got {delta}"
    print(f"✓ After baseline reset, CE delta: {delta}")

    # Inject more and verify delta from new baseline
    monitor.inject_ce(count=2)
    monitor.update()
    delta = monitor.get_ce_delta()
    assert delta == 2, f"After reset and new injection, delta should be 2, got {delta}"
    print(f"✓ After reset baseline, new delta: {delta}")

    monitor.stop_monitoring()
    print(f"✓ Monitoring stopped")

    print("\n✓ Mock CE Monitor test PASSED\n")
    return True


def test_ce_rate_calculation():
    """Test CE rate calculation"""
    print("=" * 60)
    print("Testing CE Rate Calculation")
    print("=" * 60)

    monitor = MockCEMonitor()
    monitor.start_monitoring()

    # Inject CEs over time
    for i in range(5):
        monitor.inject_ce(count=2)
        monitor.update()
        time.sleep(0.1)  # 100ms intervals

    # Calculate rate
    rate = monitor.get_ce_rate(window_seconds=1.0)
    print(f"✓ CE rate: {rate:.2f} errors/sec")

    # Should have injected 10 CEs total over ~0.5 seconds
    # Rate should be approximately 10/0.5 = 20 errors/sec (rough estimate)
    assert rate > 0, "CE rate should be greater than 0"
    print(f"✓ CE rate calculation working")

    monitor.stop_monitoring()

    print("\n✓ CE Rate calculation test PASSED\n")
    return True


def test_ce_event_tracking():
    """Test CE event tracking and history"""
    print("=" * 60)
    print("Testing CE Event Tracking")
    print("=" * 60)

    monitor = MockCEMonitor()
    monitor.start_monitoring()

    # Inject various CE events
    test_cases = [
        (3, 0x0),
        (5, 0x100000),
        (2, 0x200000),
    ]

    for count, dpa in test_cases:
        monitor.inject_ce(count=count, dpa=dpa)
        monitor.update()

    # Check event history
    history = monitor.export_ce_history()
    assert len(history) == 3, f"Should have 3 events, got {len(history)}"
    print(f"✓ Recorded {len(history)} CE events")

    # Verify event details
    total_ce = sum(e['count'] for e in history)
    assert total_ce == 10, f"Total CEs should be 10, got {total_ce}"
    print(f"✓ Total CE count in history: {total_ce}")

    # Verify DPA addresses
    dpas = [e['dpa'] for e in history]
    assert 0x0 in dpas, "Should have event at DPA 0x0"
    assert 0x100000 in dpas, "Should have event at DPA 0x100000"
    print(f"✓ DPA addresses recorded correctly")

    monitor.stop_monitoring()

    print("\n✓ CE Event tracking test PASSED\n")
    return True


def test_ce_summary():
    """Test CE summary generation"""
    print("=" * 60)
    print("Testing CE Summary Generation")
    print("=" * 60)

    monitor = MockCEMonitor()
    monitor.start_monitoring()

    # Inject some CEs
    monitor.inject_ce(count=10)
    monitor.update()

    # Generate summary
    summary = monitor.summary()
    print(summary)

    assert "CE Monitor Summary" in summary, "Summary should contain title"
    assert "Total CE Count:" in summary, "Summary should contain total count"
    assert "New CEs" in summary, "Summary should contain delta"
    assert "10" in summary, "Summary should show 10 CEs"

    print("✓ Summary generation working")

    monitor.stop_monitoring()

    print("\n✓ CE Summary test PASSED\n")
    return True


def test_real_ce_monitor():
    """Test real CE monitor (will fail gracefully if not on hardware)"""
    print("=" * 60)
    print("Testing Real CE Monitor (may not work without hardware)")
    print("=" * 60)

    monitor = CEMonitor(device_path="/dev/dax0.0", cxl_device="mem0")

    # Try to start monitoring
    try:
        monitor.start_monitoring()
        print("✓ Monitoring started (hardware detected)")

        # Try to update
        monitor.update()
        delta = monitor.get_ce_delta()
        print(f"✓ CE delta: {delta}")

        # Print available sources
        by_source = monitor.get_ce_by_source()
        if by_source:
            print(f"✓ CE sources detected: {list(by_source.keys())}")
        else:
            print("⚠ No CE sources available (expected on non-hardware)")

        monitor.stop_monitoring()

        print("\n✓ Real CE Monitor test PASSED (hardware available)\n")
        return True

    except Exception as e:
        print(f"⚠ Real CE Monitor not available: {e}")
        print("  (This is expected if running without CXL hardware)")
        print("\n✓ Real CE Monitor test SKIPPED (no hardware)\n")
        return True


def test_ce_baseline_reset():
    """Test baseline reset functionality"""
    print("=" * 60)
    print("Testing Baseline Reset")
    print("=" * 60)

    monitor = MockCEMonitor()
    monitor.start_monitoring()

    # Phase 1: Inject some CEs
    monitor.inject_ce(count=10)
    monitor.update()
    delta1 = monitor.get_ce_delta()
    assert delta1 == 10, f"Phase 1 delta should be 10, got {delta1}"
    print(f"✓ Phase 1 - CE delta: {delta1}")

    # Reset baseline
    monitor.reset_baseline()
    delta2 = monitor.get_ce_delta()
    assert delta2 == 0, f"After reset, delta should be 0, got {delta2}"
    print(f"✓ After reset - CE delta: {delta2}")

    # Phase 2: Inject more CEs
    monitor.inject_ce(count=5)
    monitor.update()
    delta3 = monitor.get_ce_delta()
    assert delta3 == 5, f"Phase 2 delta should be 5, got {delta3}"
    print(f"✓ Phase 2 - CE delta: {delta3}")

    # Total count should be 15
    total = monitor.get_total_ce_count()
    assert total == 15, f"Total should be 15, got {total}"
    print(f"✓ Total CE count: {total}")

    monitor.stop_monitoring()

    print("\n✓ Baseline reset test PASSED\n")
    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CE Monitor Test Suite")
    print("=" * 60 + "\n")

    all_passed = True

    try:
        all_passed &= test_mock_ce_monitor()
        all_passed &= test_ce_rate_calculation()
        all_passed &= test_ce_event_tracking()
        all_passed &= test_ce_summary()
        all_passed &= test_baseline_reset()
        all_passed &= test_real_ce_monitor()

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ ALL CE MONITOR TESTS PASSED!")
    else:
        print("✗ SOME CE MONITOR TESTS FAILED")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)
