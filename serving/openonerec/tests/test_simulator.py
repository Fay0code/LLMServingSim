"""Feature-contract regression tests; synthetic clocks are not accuracy evidence."""
import unittest
import random
from serving.openonerec.state import RecRequest, UserState, Directory
from serving.openonerec.simulator import Simulator
from serving.openonerec.timing import SyntheticTiming, AstraTiming, build_stage_graph


def req(i, user='u', at=0, version='v1', tokens=(1,2,3), model='m1', budget=0):
    return RecRequest(str(i), user, model, version, tokens, at, budget)


def sim(**kw):
    return Simulator(kw.pop('gpu_capacities', {'g0': 1000}), kw.pop('timing', SyntheticTiming()),
                     beam_width=2, kv_bytes_per_token=10, **kw)


class Contracts(unittest.TestCase):
    def test_actual_prefix_versions(self):
        r=req(0)
        for other in [req(1,tokens=(1,2,4)), req(1,version='v2'), req(1,model='m2'),req(1,user='v')]:
            self.assertNotEqual(r.prefix_key,other.prefix_key)
        self.assertEqual(r.prefix_key,req(1).prefix_key)

    def test_invalid_requests(self):
        for kwargs in [dict(at=-1),dict(tokens=(-1,)),dict(user=''),dict(budget=True)]:
            with self.assertRaises(ValueError): req(0,**kwargs)

    def test_arrivals_not_dependency(self):
        r=sim(gpu_capacities={'g0':1000,'g1':1000}).run([req(0),req(1,at=10)])
        responses={x['request_id']:x for x in r['responses']}
        self.assertEqual(responses['1']['arrival_time_ns'],10)
        self.assertEqual(responses['1']['waiting_ns'],0)
        self.assertLess(responses['1']['arrival_time_ns'],responses['0']['response_time_ns'])

    def test_version_change_no_hit(self):
        r=sim().run([req(0),req(1,at=10000,version='v2')])
        self.assertEqual(r['responses'][1]['mode'],'recompute')

    def test_retained_prefix(self):
        r=sim().run([req(0),req(1,at=10000)])
        self.assertEqual(r['responses'][1]['mode'],'local')
        self.assertEqual([e['stage'] for e in r['events'] if e['event']=='batch_start'],[0,1,2,0,1,2])

    def test_writeback_not_response(self):
        t=SyntheticTiming(transfer_base_ns=20000)
        s=sim(timing=t,cpu_capacity_bytes=1000)
        r=s.run([req(0),req(1,at=7000)])
        writes=[e for e in r['events'] if e['event']=='writeback_complete']
        self.assertEqual(len(writes),1)
        self.assertGreater(writes[0]['time_ns'],r['responses'][0]['response_time_ns'])
        self.assertGreater(r['drained_time_ns'],max(x['response_time_ns'] for x in r['responses']))
        self.assertTrue(all(c.ready and c.pins==0 for c in s.directory.copies.values()))

    def test_pending_is_not_hit(self):
        users={'u':UserState()}; users['u'].observe(0)
        d=Directory({'cpu':100},users,1000,0)
        c=d.reserve(req(0).prefix_key,'cpu',30,0)
        self.assertIsNone(d.valid(c.key,'cpu',10))
        self.assertFalse(d.make_room('cpu',80,10))
        c.ready=True
        self.assertTrue(d.make_room('cpu',80,10))

    def test_cpu_reload_and_source_recency(self):
        t=SyntheticTiming(stage_base_ns=1, query_ns=0, attention_token_ns=0,
                          prefix_token_ns=1000,transfer_base_ns=0,bytes_per_ns=100)
        s=sim(timing=t,cpu_capacity_bytes=1000,ttl_ns=100)
        s.users['u']=UserState(); s.users['u'].observe(0)
        c=s.directory.reserve(req(0).prefix_key,'cpu',30,0);c.ready=True
        r=s.run([req(0,at=50)])
        self.assertEqual(r['responses'][0]['mode'],'cpu_reload')
        self.assertEqual(c.last_used_ns,51)
        self.assertIs(s.directory.valid(c.key,'cpu',120),c)

    def test_memory_peak_and_free(self):
        s=sim(gpu_capacities={'g0':180},max_batch_size=4)
        r=s.run([req(i,user=str(i)) for i in range(10)])
        self.assertEqual(len(r['responses']),10)
        self.assertLessEqual(r['peak_bytes']['g0'],180)
        self.assertEqual(s.directory.workspace['g0'],0)
        self.assertTrue(all(c.pins==0 for c in s.directory.copies.values()))

    def test_cold_fcfs_admission(self):
        s=sim(gpu_capacities={'g0':90},max_batch_size=1)
        r=s.run([req(0,user='cold'),req(1,user='hot'),req(2,user='hot')])
        self.assertEqual([x['request_id'] for x in r['responses']],['0','1','2'])

    def test_stage_budget_urgency(self):
        r=sim(max_batch_size=1).run([req(0,user='no-budget'),req(1,user='urgent',budget=100)])
        starts=[e for e in r['events'] if e['event']=='batch_start']
        self.assertEqual(starts[0]['request_ids'],['1'])
        self.assertTrue(next(x for x in r['responses'] if x['request_id']=='1')['deadline_missed'])

    def test_aging_beats_budget(self):
        r=sim(max_batch_size=1,aging_threshold_ns=1).run([req(0,user='no-budget'),req(1,user='urgent',budget=100)])
        starts=[e for e in r['events'] if e['event']=='batch_start']
        self.assertEqual(starts[0]['request_ids'],['1'])
        self.assertEqual(starts[1]['request_ids'],['0'])

    def test_no_future_observations(self):
        short=sim().run([req(0),req(1,at=1)])
        long=sim().run([req(0),req(1,at=1),req(2,at=100000)])
        self.assertEqual(short['events'],[e for e in long['events'] if e['time_ns']<100000])

    def test_rate_decay(self):
        state=UserState(); state.observe(0);state.observe(1000)
        self.assertEqual(state.rate_at(2000,1000),state.arrival_rate_hz/2)
        with self.assertRaises(ValueError):state.observe(1)

    def test_expired_version_copy(self):
        r=sim(ttl_ns=100).run([req(0),req(1,at=10000)])
        self.assertEqual(r['responses'][1]['mode'],'recompute')

    def test_rejected_oversize_duplicate_and_rerun(self):
        with self.assertRaises(ValueError):sim(gpu_capacities={'g0':10}).run([req(0)])
        with self.assertRaises(ValueError):sim().run([req(0),req(0)])
        s=sim();s.run([])
        with self.assertRaises(ValueError):s.run([])

    def test_zero_time_terminates(self):
        t=SyntheticTiming(stage_base_ns=0,query_ns=0,prefix_token_ns=0,attention_token_ns=0,transfer_base_ns=0)
        r=sim(timing=t).run([req(i) for i in range(20)])
        self.assertEqual(len(r['responses']),20)
        self.assertEqual(r['drained_time_ns'],0)

    def test_estimates_never_execute(self):
        calls=[]
        t=AstraTiming(lambda d,g: calls.append(('stage',g.stage)) or 1,
                     lambda s,d,n: calls.append(('transfer',s,d)) or 1,
                     estimate_stage=lambda d,g: 1,estimate_transfer=lambda s,d,n:1)
        sim(timing=t).run([req(0)])
        self.assertEqual(calls,[('stage',0),('stage',1),('stage',2)])

    def test_randomized_lifecycle_invariants(self):
        for seed in range(100):
            rng = random.Random(seed)
            requests = [RecRequest(str(i), str(rng.randrange(4)), 'v1', 'p1',
                                   tuple(range(rng.choice([3, 6, 9]))),
                                   rng.randrange(20000), rng.choice([0, 1000, 10000]))
                        for i in range(50)]
            simulator = sim(gpu_capacities={'g0': 180, 'g1': 200},
                            timing=SyntheticTiming(prefix_token_ns=1000, bytes_per_ns=1),
                            cpu_capacity_bytes=100, max_batch_size=4,
                            ttl_ns=5000, aging_threshold_ns=3000)
            result = simulator.run(requests)
            self.assertEqual(len(result['responses']), 50)
            for location, capacity in simulator.directory.capacities.items():
                self.assertLessEqual(result['peak_bytes'][location], capacity)
            self.assertTrue(all(value == 0 for value in simulator.directory.workspace.values()))
            self.assertTrue(all(copy.pins == 0 and copy.ready
                                for copy in simulator.directory.copies.values()))

    def test_graph_shapes_and_bad_timing(self):
        for stage in range(3):
            g=build_stage_graph(stage,[3,4],2,['recompute','local'])
            self.assertEqual(g.query_tokens,(1,1) if stage==0 else (2,2))
            self.assertEqual(g.prefix_prefill_tokens,(3,0) if stage==0 else (0,0))
            self.assertFalse(g.describe()['executable_astra_graph'])
        with self.assertRaises(ValueError):build_stage_graph(3,[3],2,['local'])
        t=AstraTiming(lambda d,g:float('nan'),lambda s,d,n:1,
                     estimate_stage=lambda d,g:1,estimate_transfer=lambda s,d,n:1)
        with self.assertRaises(ValueError):sim(timing=t).run([req(0)])

if __name__=='__main__': unittest.main(verbosity=2)
