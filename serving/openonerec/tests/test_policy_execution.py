"""Policy action contracts for bounded waves and nonpreemptive fixed steps."""
from dataclasses import replace
import unittest

from serving.openonerec.policy import AcquireAction, BatchAction, IllegalAction, RouteAction, WaitAction
from serving.openonerec.tests.test_policy import environment, request, run


class PolicyExecutionContracts(unittest.TestCase):
    def ready(self, env, requests):
        env.reset(requests)
        for r in requests:
            env.step(RouteAction(r.request_id, 'g'))
            env.step(AcquireAction(r.request_id, 'recompute'))

    def test_wave_accepts_mixed_stages_at_boundaries(self):
        env = environment(execution_mode='wave')
        self.ready(env, [request(0), request(1)])
        env.step(BatchAction('g', ('0',)))
        with self.assertRaises(IllegalAction):
            env.step(BatchAction('g', ('1',)))
        env.step(WaitAction())
        self.assertEqual([r.stage for r in env.observe().requests], [1, 0])
        estimate = env.estimate(BatchAction('g', ('0', '1')))
        env.step(BatchAction('g', ('0', '1')))
        self.assertEqual(env._sim.records[-1]['duration_ns'], estimate)
        env.step(WaitAction())
        self.assertEqual([r.stage for r in env.observe().requests], [2, 1])

    def test_fixed_step_group_cannot_insert_arrival(self):
        env = environment(execution_mode='fixed_step')
        env.reset([request(0), request(1, at=1)])
        env.step(RouteAction('0', 'g'))
        env.step(AcquireAction('0', 'recompute'))
        duration = env.estimate(BatchAction('g', ('0',)))
        env.step(BatchAction('g', ('0',)))
        env.step(WaitAction())
        self.assertEqual(env.observe().time_ns, 1)
        env.step(RouteAction('1', 'g'))
        env.step(AcquireAction('1', 'recompute'))
        with self.assertRaises(IllegalAction):
            env.step(BatchAction('g', ('1',)))
        env.step(WaitAction())
        self.assertEqual(env.observe().time_ns, duration)
        self.assertEqual(env.observe().completed_request_ids, ('0',))
        self.assertEqual(len([e for e in env._sim.records if e['event'] == 'batch_start']), 1)

    def test_resource_masks_are_atomic_and_visible(self):
        for options in ({'max_batch_rows': 3}, {'max_batch_tokens': 18}):
            env = environment(**options)
            self.ready(env, [request(0), request(1)])
            before = env.observe()
            self.assertIn(BatchAction('g', ('0',)), before.legal_actions)
            with self.assertRaises(IllegalAction):
                env.step(BatchAction('g', ('0', '1')))
            self.assertEqual(before, env.observe())
            for name, value in options.items():
                self.assertEqual(getattr(before, name), value)

    def test_active_admission_mask_and_liveness(self):
        env = environment(max_active_requests=1, execution_mode='fixed_step')
        env.reset([request(0), request(1)])
        for i in ('0', '1'):
            env.step(RouteAction(i, 'g'))
        env.step(AcquireAction('0', 'recompute'))
        self.assertNotIn(AcquireAction('1', 'recompute'), env.observe().legal_actions)
        with self.assertRaises(IllegalAction):
            env.step(AcquireAction('1', 'recompute'))
        env.step(BatchAction('g', ('0',)))
        env.step(WaitAction())
        self.assertIn(AcquireAction('1', 'recompute'), env.observe().legal_actions)
        for mode in ('stage', 'wave', 'fixed_step'):
            result = run(environment(execution_mode=mode, max_active_requests=1,
                                     max_batch_rows=2), [request(i) for i in range(6)])
            self.assertEqual(len(result['responses']), 6)

    def test_variable_beams_validate_memory_and_singleton_budget(self):
        wide = replace(request(0), beam_rows=4)
        with self.assertRaises(ValueError):
            environment(max_batch_rows=3).reset([wide])
        with self.assertRaises(ValueError):
            environment(capacities={'g': 100}).reset([wide])
        env = environment(capacities={'g': 100, 'h': 200})
        obs = env.reset([wide])
        self.assertNotIn(RouteAction('0', 'g'), obs.legal_actions)
        self.assertIn(RouteAction('0', 'h'), obs.legal_actions)

    def test_partial_remote_estimate_and_observation(self):
        env = environment(capacities={'g': 1000, 'h': 1000})
        full = request(0, tokens=(1, 2, 3, 4))
        prefix = request('prefix', tokens=(1, 2))
        env.reset([full])
        copy = env._sim.directory.reserve(prefix.prefix_key, 'h', 20, 0,
                                          prefix_token_ids=prefix.prefix_token_ids)
        copy.ready = True
        env.step(RouteAction('0', 'g'))
        action = AcquireAction('0', 'remote', 'h')
        self.assertIn(action, env.observe().legal_actions)
        self.assertEqual(env.estimate(action),
                         env._sim.timing.estimate_transfer_ns('h', 'g', 20))
        obs = env.step(action).observation
        self.assertEqual(obs.requests[0].cached_prefix_tokens, 2)
        self.assertEqual(obs.copies[0].prefix_token_ids, (1, 2))
        self.assertFalse(obs.requests[0].ready)
        env.step(WaitAction())
        self.assertIsNone(env._sim.directory.valid(full.prefix_key, 'g', env._sim.now))
        env.step(BatchAction('g', ('0',)))
        env.step(WaitAction())
        self.assertIsNotNone(env._sim.directory.valid(full.prefix_key, 'g', env._sim.now))

    def test_partial_local_memory_rejection_is_atomic(self):
        env = environment(capacities={'g': 100})
        full = request(0, tokens=(1, 2, 3, 4))
        prefix = request('prefix', tokens=(1, 2))
        env.reset([full])
        copy = env._sim.directory.reserve(prefix.prefix_key, 'g', 20, 0,
                                          prefix_token_ids=prefix.prefix_token_ids)
        copy.ready = True
        env.step(RouteAction('0', 'g'))
        before = env.observe()
        self.assertNotIn(AcquireAction('0', 'local', 'g'), before.legal_actions)
        with self.assertRaises(IllegalAction):
            env.step(AcquireAction('0', 'local', 'g'))
        self.assertEqual(env.observe(), before)
        self.assertIn(AcquireAction('0', 'recompute'), before.legal_actions)

    def test_new_execution_modes_do_not_reveal_future_requests(self):
        for mode in ('wave', 'fixed_step'):
            a, b = environment(execution_mode=mode), environment(execution_mode=mode)
            self.assertEqual(a.reset([request(0)]),
                             b.reset([request(0), request(1, at=10000)]))


if __name__ == '__main__':
    unittest.main()
