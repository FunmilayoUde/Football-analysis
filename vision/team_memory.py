from __future__ import annotations

from collections import Counter, deque


class TeamMemory:
    def __init__(
        self,
        history_size: int = 30,
        min_votes: int = 4,
        lock_min_votes: int = 8,
        lock_ratio: float = 0.70,
    ):
        self.history_size = int(history_size)
        self.min_votes = int(min_votes)
        self.lock_min_votes = int(lock_min_votes)
        self.lock_ratio = float(lock_ratio)
        self._votes: dict[int, deque[int]] = {}
        self._locked: dict[int, int] = {}

    def update(self, track_id: int, team_id: int) -> None:
        if track_id < 0 or team_id < 0:
            return
        if track_id not in self._votes:
            self._votes[track_id] = deque(maxlen=self.history_size)
        self._votes[track_id].append(int(team_id))

        if track_id not in self._locked:
            votes = self._votes[track_id]
            if len(votes) >= self.lock_min_votes:
                top_team, top_count = Counter(votes).most_common(1)[0]
                if (top_count / max(len(votes), 1)) >= self.lock_ratio:
                    self._locked[track_id] = int(top_team)

    def get(self, track_id: int) -> int:
        if track_id in self._locked:
            return self._locked[track_id]
        votes = self._votes.get(track_id)
        if not votes or len(votes) < self.min_votes:
            return -1
        return Counter(votes).most_common(1)[0][0]
