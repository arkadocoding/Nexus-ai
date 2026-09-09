"""
Tests for the NEXUS recovery policy.
"""

from app.recovery import RecoveryPolicy


def test_recovery_allows_retry_when_recoverable() -> None:
    """Recoverable failures should allow another attempt."""

    policy = RecoveryPolicy(
        max_retries=2
    )

    assert policy.can_retry(
        retry_count=0,
        recoverable=True,
    ) is True


def test_recovery_allows_second_retry() -> None:
    """Recovery should allow the final configured retry."""

    policy = RecoveryPolicy(
        max_retries=2
    )

    assert policy.can_retry(
        retry_count=1,
        recoverable=True,
    ) is True


def test_recovery_stops_at_retry_limit() -> None:
    """Recovery must stop after the retry limit."""

    policy = RecoveryPolicy(
        max_retries=2
    )

    assert policy.can_retry(
        retry_count=2,
        recoverable=True,
    ) is False


def test_recovery_rejects_non_recoverable_failure() -> None:
    """Non-recoverable failures must never be retried."""

    policy = RecoveryPolicy(
        max_retries=2
    )

    assert policy.can_retry(
        retry_count=0,
        recoverable=False,
    ) is False


def test_recovery_never_allows_retry_after_limit() -> None:
    """Recovery must remain blocked beyond the retry limit."""

    policy = RecoveryPolicy(
        max_retries=2
    )

    assert policy.can_retry(
        retry_count=3,
        recoverable=True,
    ) is False
