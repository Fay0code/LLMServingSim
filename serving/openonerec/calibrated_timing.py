"""Validated measured-duration lookup, independent of GPU and ASTRA runtimes.

This reader preserves acquisition provenance; it cannot certify measurements.
Interpolation is opt-in, bounded, and never turns fixture data into measurements.
No timing is inferred for an unmeasured deployment, stage, beam or cache state.
"""
import itertools
import json
from fractions import Fraction
from pathlib import Path

from .timing import StageGraph, _endpoint, _integer


class MissingProfileError(LookupError):
    """The requested execution shape is not covered by this profile."""


def _keys(value, required, name, optional=()):
    if (not isinstance(value, dict) or not set(required) <= set(value)
            or set(value) - set(required) - set(optional)):
        raise ValueError(f'{name} must contain exactly {sorted(required)}')


def _bracket(values, target):
    lower = [x for x in values if x <= target]
    upper = [x for x in values if x >= target]
    if not lower or not upper:
        raise MissingProfileError(f'coordinate {target} is outside measured coverage')
    lo, hi = max(lower), min(upper)
    return ((lo, Fraction(1)),) if lo == hi else (
        (lo, Fraction(hi - target, hi - lo)),
        (hi, Fraction(target - lo, hi - lo)),
    )


def _interpolate(points, target):
    axes = [_bracket({point[i] for point in points}, coordinate)
            for i, coordinate in enumerate(target)]
    value = Fraction(0)
    for corner in itertools.product(*axes):
        key = tuple(x[0] for x in corner)
        if key not in points:
            raise MissingProfileError(f'missing interpolation corner {key}')
        weight = Fraction(1)
        for _, factor in corner:
            weight *= factor
        value += points[key] * weight
    # Durations are conservative integral nanoseconds, with no float roundoff.
    return (value.numerator + value.denominator - 1) // value.denominator


class CalibratedTiming:
    """Timing protocol backed by explicit, deployment-specific JSON samples.

    Stage rows specify the complete ordered prefix-length and acquisition-mode
    vectors. Only stage-0 ``recompute`` differs from ``ready``: other acquisition
    must finish before forward execution. At later stages all prefixes are ready.
    Exact heterogeneous batches are supported without averaging their histories.
    Optional interpolation uses homogeneous history and batch-size axes only;
    stage, destination, beam width, mode and partial cached length are exact categories.
    Transfers are directed, and interpolate only on byte count.

    ``data_kind='fixture'`` supports schema tests/examples and is visibly synthetic.
    ``data_kind='measured'`` is the producer's provenance declaration, not proof
    of hardware measurement or end-to-end simulator accuracy.
    """

    def __init__(self, document, *, interpolate=False):
        if not isinstance(interpolate, bool):
            raise ValueError('interpolate must be boolean')
        _keys(document, ('schema_version', 'data_kind', 'provenance', 'stages', 'transfers'), 'profile')
        if type(document['schema_version']) is not int or document['schema_version'] != 1:
            raise ValueError('schema_version must be 1')
        if document['data_kind'] not in ('measured', 'fixture'):
            raise ValueError('data_kind must be measured or fixture')
        _keys(document['provenance'], ('model', 'hardware', 'software', 'measurement_source'), 'provenance')
        for key, value in document['provenance'].items():
            _endpoint(value, f'provenance.{key}')
        self.provenance = dict(document['provenance'])
        self.data_kind = document['data_kind']
        self.label = ('measured-profile' if self.data_kind == 'measured'
                      else 'synthetic-profile-fixture')
        self.interpolate = interpolate
        self._stages = {}
        self._homogeneous = {}
        self._transfers = {}
        for section in ('stages', 'transfers'):
            if not isinstance(document[section], list):
                raise ValueError(f'{section} must be a list')
        if not document['stages']:
            raise ValueError('stages must contain at least one measurement')
        for row in document['stages']:
            _keys(row, ('destination', 'stage', 'batch_size', 'beam_width',
                        'prefix_tokens', 'kv_mode', 'duration_ns'), 'stage row',
                  optional=('cached_prefix_tokens',))
            _endpoint(row['destination'], 'destination')
            if not isinstance(row['prefix_tokens'], list) or not isinstance(row['kv_mode'], list):
                raise ValueError('prefix_tokens and kv_mode must be JSON arrays')
            stage = _integer(row['stage'], 'stage')
            beam = _integer(row['beam_width'], 'beam_width', 1)
            batch = _integer(row['batch_size'], 'batch_size', 1)
            if any(mode not in ('ready', 'recompute') for mode in row['kv_mode']):
                raise ValueError('profile kv_mode entries must be ready or recompute')
            if stage != 0 and 'recompute' in row['kv_mode']:
                raise ValueError('recompute applies only to stage 0')
            cached = row.get('cached_prefix_tokens')
            if 'cached_prefix_tokens' in row and not isinstance(cached, list):
                raise ValueError('cached_prefix_tokens must be a JSON array')
            extra = {} if cached is None else {'cached_prefix_tokens': tuple(cached)}
            graph = StageGraph(stage, batch, beam, tuple(row['prefix_tokens']),
                               (1 if stage == 0 else beam,) * batch, tuple(row['kv_mode']), **extra)
            duration = _integer(row['duration_ns'], 'duration_ns')
            key = self._key(row['destination'], graph)
            if key in self._stages:
                raise ValueError(f'duplicate stage shape {key}')
            self._stages[key] = duration
            cache_categories = self._cache_categories(graph)
            if (len(set(graph.prefix_tokens)) == len(set(key[4]))
                    == len(set(cache_categories)) == 1):
                category = (row['destination'], stage, beam, key[4][0], cache_categories[0])
                self._homogeneous.setdefault(category, {})[(graph.prefix_tokens[0], batch)] = duration
        for row in document['transfers']:
            _keys(row, ('source', 'destination', 'bytes_count', 'duration_ns'), 'transfer row')
            for name in ('source', 'destination'):
                _endpoint(row[name], name)
            size = _integer(row['bytes_count'], 'bytes_count', 1)
            duration = _integer(row['duration_ns'], 'duration_ns')
            if row['source'] == row['destination']:
                raise ValueError('transfer measurements must have distinct endpoints')
            points = self._transfers.setdefault((row['source'], row['destination']), {})
            if (size,) in points:
                raise ValueError('duplicate transfer size for directed endpoints')
            points[(size,)] = duration

    @classmethod
    def from_json(cls, path, *, interpolate=False):
        """Read a profile file; reject duplicate object fields and nonfinite JSON."""
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f'duplicate JSON field {key}')
                result[key] = value
            return result

        def invalid(value):
            raise ValueError(f'nonfinite JSON value {value}')

        document = json.loads(Path(path).read_text(), object_pairs_hook=unique,
                              parse_constant=invalid)
        return cls(document, interpolate=interpolate)

    @staticmethod
    def _key(destination, graph):
        # Acquisition has already completed, except stage-0 recomputation.
        modes = tuple('recompute' if graph.stage == 0 and mode == 'recompute' else 'ready'
                      for mode in graph.kv_mode)
        cached = tuple(prefix - missing for prefix, missing in
                       zip(graph.prefix_tokens, graph.prefix_prefill_tokens))
        return (destination, graph.stage, graph.beam_width, graph.prefix_tokens, modes, cached)

    @staticmethod
    def _cache_categories(graph):
        return tuple('all' if missing == 0 else prefix - missing
                     for prefix, missing in zip(graph.prefix_tokens, graph.prefix_prefill_tokens))

    def duration_ns(self, destination, graph):
        _endpoint(destination, 'destination')
        if not isinstance(graph, StageGraph):
            raise TypeError('graph must be a StageGraph')
        key = self._key(destination, graph)
        if key in self._stages:
            return self._stages[key]
        modes = key[4]
        cache_categories = self._cache_categories(graph)
        if (self.interpolate and len(set(graph.prefix_tokens)) == len(set(modes))
                == len(set(cache_categories)) == 1):
            category = (destination, graph.stage, graph.beam_width, modes[0], cache_categories[0])
            points = self._homogeneous.get(category, {})
            if points:
                return _interpolate(points, (graph.prefix_tokens[0], graph.batch_size))
        raise MissingProfileError(f'no measured stage coverage for {key}')

    def transfer_ns(self, source, destination, bytes_count):
        _endpoint(source, 'source')
        _endpoint(destination, 'destination')
        _integer(bytes_count, 'bytes_count')
        if source == destination or bytes_count == 0:
            return 0
        points = self._transfers.get((source, destination), {})
        if (bytes_count,) in points:
            return points[(bytes_count,)]
        if self.interpolate and points:
            return _interpolate(points, (bytes_count,))
        raise MissingProfileError(f'no measured transfer coverage for {source}->{destination}, {bytes_count} bytes')

    # Lookup is pure; admission estimates cannot advance an external backend.
    estimate_duration_ns = duration_ns
    estimate_transfer_ns = transfer_ns
