"""Durable policy-boundary contracts; no calibrated performance assertions."""
from dataclasses import FrozenInstanceError
import unittest

from serving.openonerec.policy import (AcquireAction, BatchAction, FCFSReferencePolicy,
    IllegalAction, PolicyEnvironment, RouteAction, WaitAction)
from serving.openonerec.state import Directory, RecRequest, UserState
from serving.openonerec.timing import AstraTiming, SyntheticTiming


def request(i, at=0, user='u', tokens=(1, 2, 3)):
    return RecRequest(str(i), user, 'm', 'p', tokens, at)


def environment(**kwargs):
    return PolicyEnvironment(kwargs.pop('capacities', {'g': 1000}),
                             kwargs.pop('timing', SyntheticTiming()),
                             beam_width=2, kv_bytes_per_token=10, **kwargs)


def run(env, requests, seed=0):
    policy = FCFSReferencePolicy()
    obs = env.reset(requests, seed=seed)
    for _ in range(10000):
        if obs.done:
            return env.result()
        obs = env.step(policy(obs)).observation
    raise AssertionError('episode did not terminate')


class PolicyContracts(unittest.TestCase):
    def test_snapshot_immutable_and_causal(self):
        a, b = environment(), environment()
        oa = a.reset([request(0)])
        ob = b.reset([request(0), request(99, 20000, 'future')])
        self.assertEqual(oa, ob)
        with self.assertRaises(FrozenInstanceError):
            oa.time_ns = 5
        with self.assertRaises(FrozenInstanceError):
            oa.requests[0].request.user_id = 'changed'
        a.step(RouteAction('0', 'g'))
        self.assertIsNone(oa.requests[0].destination)

    def test_invalid_actions_are_atomic(self):
        env = environment()
        obs = env.reset([request(0), request(1)])
        for action in [RouteAction('future', 'g'), AcquireAction('0', 'local', 'g'),
                       BatchAction('g', ('0',)), WaitAction(0), WaitAction(True),
                       RouteAction([], 'g'), BatchAction('g', ([1],))]:
            with self.assertRaises(IllegalAction):
                env.step(action)
            self.assertEqual(env.observe(), obs)
        env.step(RouteAction('0', 'g'))
        env.step(RouteAction('1', 'g'))
        with self.assertRaises(IllegalAction):
            env.step(AcquireAction('1', 'recompute'))
        with self.assertRaises(IllegalAction):
            env.step(AcquireAction('0', 'remote', 'cpu'))

    def test_explicit_same_stage_batch_and_busy_guard(self):
        env = environment()
        env.reset([request(0), request(1)])
        for i in ('0', '1'):
            env.step(RouteAction(i, 'g'))
            env.step(AcquireAction(i, 'recompute'))
        env.step(BatchAction('g', ('0',)))
        with self.assertRaises(IllegalAction):
            env.step(BatchAction('g', ('1',)))
        env.step(WaitAction())
        with self.assertRaises(IllegalAction):
            env.step(BatchAction('g', ('0', '1')))
        env.step(BatchAction('g', ('1',)))
        env.step(WaitAction())
        env.step(BatchAction('g', ('0', '1')))
        self.assertEqual(env._sim.records[-1]['request_ids'], ['0', '1'])

    def test_wait_timer_and_aging_boundary(self):
        env = environment(aging_threshold_ns=10, max_wait_ns=100)
        env.reset([request(0), request(1, 1000)])
        self.assertEqual(env.step(WaitAction(50)).observation.time_ns, 10)
        with self.assertRaises(IllegalAction):
            env.step(WaitAction(20))
        env.step(RouteAction('0', 'g'))
        env.step(AcquireAction('0', 'recompute'))
        env.step(BatchAction('g', ('0',)))
        self.assertEqual(env.step(WaitAction(20)).observation.time_ns, 20)

    def test_timer_without_external_events(self):
        env = environment(aging_threshold_ns=10)
        env.reset([request(0)])
        self.assertEqual(env.step(WaitAction(5)).observation.time_ns, 5)
        self.assertEqual(env.step(WaitAction(100)).observation.time_ns, 10)
        with self.assertRaises(IllegalAction):
            env.step(WaitAction(11))

    def test_estimation_never_executes(self):
        calls = []
        timing = AstraTiming(lambda d, g: calls.append(('stage', g.stage)) or 1,
                             lambda s, d, n: calls.append(('transfer', n)) or 1,
                             estimate_stage=lambda d, g: 2, estimate_transfer=lambda s, d, n: 3)
        env = environment(timing=timing)
        env.reset([request(0)])
        self.assertEqual(env.estimate(RouteAction('0', 'g')), 6)
        env.step(RouteAction('0', 'g'))
        env.step(AcquireAction('0', 'recompute'))
        self.assertEqual(env.estimate(BatchAction('g', ('0',))), 2)
        env.observe()
        self.assertEqual(calls, [])
        env.step(BatchAction('g', ('0',)))
        self.assertEqual(calls, [('stage', 0)])

    def test_directory_queries_and_admission_checks_are_pure(self):
        user = UserState(); user.observe(0)
        directory = Directory({'g': 100}, {'u': user}, 100, 10)
        copy = directory.reserve(request(0).prefix_key, 'g', 30, 0)
        copy.ready = True
        self.assertIsNone(directory.valid(copy.key, 'g', 10))
        self.assertEqual(directory.evictions, 0)
        self.assertEqual(len(directory.copies), 1)
        self.assertFalse(directory.can_fit('g', 101))
        self.assertEqual(len(directory.copies), 1)
        directory.expire(10)
        self.assertEqual(directory.evictions, 1)

    def test_reproducible_reset_and_zero_time(self):
        timing = SyntheticTiming(stage_base_ns=0, query_ns=0, prefix_token_ns=0,
                                 attention_token_ns=0, transfer_base_ns=0)
        env = environment(timing=timing)
        requests = [request(i, user=str(i)) for i in range(10)]
        a = run(env, requests, seed=42)
        self.assertEqual(a, run(env, requests, seed=42))
        self.assertEqual(len(a['responses']), 10)
        self.assertEqual(a['drained_time_ns'], 0)
        a['responses'].clear()
        self.assertEqual(len(env.result()['responses']), 10)

    def test_memory_and_source_pins(self):
        env = environment(capacities={'g': 90}, cpu_capacity_bytes=100, max_batch_size=4)
        result = run(env, [request(i, i * 100, str(i % 3)) for i in range(15)])
        self.assertEqual(len(result['responses']), 15)
        self.assertLessEqual(result['peak_bytes']['g'], 90)
        self.assertTrue(all(c.pins == 0 for c in env._sim.directory.copies.values()))
        self.assertTrue(all(n == 0 for n in env._sim.directory.workspace.values()))

    def test_remote_acquisition_pins_source_until_completion(self):
        env = environment(capacities={'g': 1000, 'h': 1000})
        obs = env.reset([request(0), request(1, 10000)])
        policy = FCFSReferencePolicy()
        while '0' not in obs.completed_request_ids:
            obs = env.step(policy(obs)).observation
        obs = env.step(WaitAction()).observation
        self.assertEqual(obs.time_ns, 10000)
        env.step(RouteAction('1', 'h'))
        obs = env.step(AcquireAction('1', 'remote', 'g')).observation
        source = next(c for c in obs.copies if c.location == 'g')
        self.assertEqual(source.pins, 1)
        self.assertFalse(obs.requests[0].ready)
        with self.assertRaises(IllegalAction):
            env.step(BatchAction('h', ('1',)))
        obs = env.step(WaitAction()).observation
        self.assertEqual(next(c for c in obs.copies if c.location == 'g').pins, 0)
        self.assertTrue(obs.requests[0].ready)
        self.assertEqual(next(c for c in obs.copies if c.location == 'h').pins, 1)

    def test_legacy_expiration_remains_key_scoped(self):
        from serving.openonerec.simulator import Simulator
        simulator = Simulator({'g': 1000}, SyntheticTiming(), beam_width=2,
                              kv_bytes_per_token=10, ttl_ns=100)
        result = simulator.run([request(0), request(1, 10000, 'v')])
        self.assertEqual(result['peak_bytes']['g'], 120)
        self.assertEqual(result['evictions'], 0)

    def test_empty_and_oversized(self):
        env = environment()
        self.assertTrue(env.reset([]).done)
        with self.assertRaises(IllegalAction):
            env.step(WaitAction())
        with self.assertRaises(ValueError):
            environment(capacities={'g': 10}).reset([request(0)])


if __name__ == '__main__':
    unittest.main()
