"""
recovery.py

Recovery policy for NEXUS.

The RecoveryPolicy decides whether NEXUS is allowed
to attempt another plan after an execution failure.

It does NOT replan or execute anything itself.
"""

from dataclasses import dataclass


@dataclass
class RecoveryPolicy:
    """
    Controls bounded recovery attempts.

    NEXUS must never retry forever.
    """

    max_retries: int = 2

    def can_retry(
        self,
        retry_count: int,
        recoverable: bool,
    ) -> bool:
        """
        Determine whether another recovery attempt
        is allowed.
        """

        if not recoverable:
            return False

        if retry_count >= self.max_retries:
            return False

        return True
