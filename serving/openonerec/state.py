"""Versioned public-prefix directory and causal user-arrival observations."""
from dataclasses import dataclass
import hashlib
import json
import math


def nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


@dataclass(frozen=True)
class RecRequest:
    request_id: str
    user_id: str
    model_version: str
    prefix_version: str
    prefix_token_ids: tuple
    arrival_time_ns: int
    budget_ns: int = 0

    def __post_init__(self):
        for name in ('request_id', 'user_id', 'model_version', 'prefix_version'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} must be a nonempty string')
        object.__setattr__(self, 'prefix_token_ids', tuple(self.prefix_token_ids))
        for token in self.prefix_token_ids:
            nonnegative_int(token, 'prefix token')
        nonnegative_int(self.arrival_time_ns, 'arrival_time_ns')
        nonnegative_int(self.budget_ns, 'budget_ns')

    @property
    def prefix_key(self):
        # Identity is not compatibility: actual token content and both versions
        # are required. User isolation also prevents cross-user private reuse.
        digest = hashlib.sha256(json.dumps(self.prefix_token_ids).encode()).hexdigest()
        return (self.user_id, self.model_version, self.prefix_version, digest)


@dataclass
class UserState:
    last_arrival_ns: int = None
    arrival_rate_hz: float = 0.0
    observations: int = 0

    def observe(self, now):
        if self.last_arrival_ns is not None:
            if now < self.last_arrival_ns:
                raise ValueError('arrival observations must be chronological')
            interval = max(1, now - self.last_arrival_ns)
            rate = 1e9 / interval
            self.arrival_rate_hz = (rate if self.observations == 1 else
                                    0.5 * rate + 0.5 * self.arrival_rate_hz)
        self.last_arrival_ns = now
        self.observations += 1

    def rate_at(self, now, half_life_ns):
        if self.last_arrival_ns is None:
            return 0.0
        return self.arrival_rate_hz * math.exp2(-max(0, now - self.last_arrival_ns) / half_life_ns)


@dataclass
class Copy:
    key: tuple
    location: str
    size_bytes: int
    ready: bool
    last_used_ns: int
    pins: int = 0


class Directory:
    """Byte-limited tiers. Pending writes reserve bytes but are not valid hits."""
    def __init__(self, capacities, users, half_life_ns, ttl_ns):
        self.capacities = capacities
        self.users = users
        self.half_life_ns = half_life_ns
        self.ttl_ns = ttl_ns
        self.copies = {}
        self.workspace = {location: 0 for location in capacities}
        self.peak = dict(self.workspace)
        self.evictions = 0

    def used(self, location):
        return self.workspace[location] + sum(c.size_bytes for c in self.copies.values()
                                                if c.location == location)

    def record_peak(self, location):
        self.peak[location] = max(self.peak[location], self.used(location))

    def valid(self, key, location, now):
        copy = self.copies.get((key, location))
        if copy is None or not copy.ready:
            return None
        if self.ttl_ns and now - copy.last_used_ns >= self.ttl_ns and not copy.pins:
            del self.copies[(key, location)]
            self.evictions += 1
            return None
        return copy

    def make_room(self, location, size_bytes, now):
        candidates = sorted(
            (c for c in self.copies.values() if c.location == location and c.ready and not c.pins),
            key=lambda c: (self.users[c.key[0]].rate_at(now, self.half_life_ns), c.last_used_ns, c.key))
        while self.used(location) + size_bytes > self.capacities[location] and candidates:
            copy = candidates.pop(0)
            del self.copies[(copy.key, location)]
            self.evictions += 1
        return self.used(location) + size_bytes <= self.capacities[location]

    def reserve(self, key, location, size_bytes, now):
        if (key, location) in self.copies:
            return None
        if not self.make_room(location, size_bytes, now):
            return None
        copy = Copy(key, location, size_bytes, False, now)
        self.copies[(key, location)] = copy
        self.record_peak(location)
        return copy
