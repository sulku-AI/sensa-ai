from sensa_ai import lookahead as la


def test_failsafe_when_analysis_behind() -> None:
    assert la.mode(10.0, 12.0, 100.0, safe_margin_s=5.0) is la.Mode.FAILSAFE
    assert la.mode(10.0, 40.0, 100.0, safe_margin_s=5.0) is la.Mode.NORMAL


def test_end_of_video_counts_as_analyzed() -> None:
    assert la.mode(98.0, 100.0, 100.0, safe_margin_s=5.0) is la.Mode.NORMAL


def test_next_window_never_behind_playhead() -> None:
    assert la.next_window(5.0, 100.0, 30.0, playhead_s=20.0) == (20.0, 50.0)
    assert la.next_window(45.0, 60.0, 30.0, playhead_s=20.0) == (45.0, 50.0)
