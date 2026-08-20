"""Client and wire contract for the problem source."""

from .api import (
    ChallengeCommitRequest,
    ChallengeCommitResponse,
    ChallengeFeedback,
    FeedbackVerdict,
    ChallengeLeaseRequest,
    ChallengeRevealRequest,
    ChallengeRevealResponse,
    MinerSubmission,
    PublicChallenge,
    derive_request_id,
)
from .client import ProblemServerClient

__all__ = [
    "ChallengeCommitRequest",
    "ChallengeCommitResponse",
    "ChallengeFeedback",
    "FeedbackVerdict",
    "ChallengeLeaseRequest",
    "ChallengeRevealRequest",
    "ChallengeRevealResponse",
    "MinerSubmission",
    "ProblemServerClient",
    "PublicChallenge",
    "derive_request_id",
]
