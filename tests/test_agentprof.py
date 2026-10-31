"""Self-tests. Stdlib only — run with: python3 tests/test_agentprof.py"""
import sys, time, unittest, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agentprof import Tracer, SpanKind, summarise, aggregate, DEFAULT_SHAPES, run_turn
from agentprof.synthetic import LatencyProfile
from agentprof.adapters.openai_like import instrument_chat, instrument_tool
from agentprof.adapters.langgraph import instrument_node, instrument_state


class TestTracer(unittest.TestCase):
    def test_blocked_time_is_wall_minus_cpu(self):
        t = Tracer()
        with t.span("turn", SpanKind.TURN):
            with t.span("llm", SpanKind.LLM_CALL):
                time.sleep(0.15)
        llm = [s for s in t.spans if s.name == "llm"][0]
        self.assertGreater(llm.wall, 0.14)
        self.assertLess(llm.cpu, 0.02)
        self.assertGreater(llm.blocked, 0.13)

    def test_unclosed_span_raises(self):
        t = Tracer()
        cm = t.span("x", SpanKind.COMPUTE)
        cm.__enter__()
        with self.assertRaises(ValueError):
            _ = t.spans[0].wall

    def test_parenting(self):
        t = Tracer()
        with t.span("turn", SpanKind.TURN):
            with t.span("step0", SpanKind.STEP):
                with t.span("llm", SpanKind.LLM_CALL):
                    pass
        by_name = {s.name: s for s in t.spans}
        self.assertIsNone(by_name["turn"].parent_id)
        self.assertEqual(by_name["step0"].parent_id, by_name["turn"].span_id)
        self.assertEqual(by_name["llm"].parent_id, by_name["step0"].span_id)

    def test_state_size_measured(self):
        t = Tracer()
        n = t.record_state({"messages": ["a" * 100]}, step=0)
        self.assertGreater(n, 100)


class TestMetrics(unittest.TestCase):
    def test_requires_exactly_one_turn(self):
        t = Tracer()
        with t.span("a", SpanKind.STEP):
            pass
        with self.assertRaises(ValueError):
            summarise(t.spans)

        t2 = Tracer()
        with t2.span("t1", SpanKind.TURN):
            pass
        with t2.span("t2", SpanKind.TURN):
            pass
        with self.assertRaises(ValueError):
            summarise(t2.spans)

    def test_idle_fraction_high_when_blocked(self):
        t = Tracer()
        with t.span("turn", SpanKind.TURN):
            with t.span("llm", SpanKind.LLM_CALL):
                time.sleep(0.2)
        m = summarise(t.spans)
        self.assertGreater(m.billed_idle_fraction, 0.9)
        self.assertGreater(m.wait_to_compute_ratio, 5)

    def test_nested_remote_spans_not_double_counted(self):
        t = Tracer()
        with t.span("turn", SpanKind.TURN):
            with t.span("outer", SpanKind.TOOL_CALL):
                with t.span("inner", SpanKind.LLM_CALL):
                    time.sleep(0.05)
        m = summarise(t.spans)
        # only the leaf counts
        self.assertNotIn("tool_call", m.blocked_by_kind_s)
        self.assertIn("llm_call", m.blocked_by_kind_s)
        self.assertTrue(any("nested" in n for n in m.notes))

    def test_state_spans_excluded_from_step_count(self):
        t = Tracer()
        with t.span("turn", SpanKind.TURN):
            with t.span("s0", SpanKind.STEP):
                t.record_state({"a": 1}, step=0)
        m = summarise(t.spans)
        self.assertEqual(m.steps, 1)
        self.assertEqual(m.remote_calls, 0)   # measured_only state span must not count


class TestSynthetic(unittest.TestCase):
    def test_shapes_run_and_have_expected_structure(self):
        for name, shape in DEFAULT_SHAPES.items():
            t = Tracer()
            run_turn(shape, t, seed=1, sleep=False)
            m = summarise(t.spans)
            self.assertEqual(m.steps, shape.steps, name)
            self.assertGreater(m.state_bytes_per_step_p50, 0, name)

    def test_seed_is_deterministic(self):
        a, b = Tracer(), Tracer()
        run_turn(DEFAULT_SHAPES["short_tool_loop"], a, seed=7, sleep=False)
        run_turn(DEFAULT_SHAPES["short_tool_loop"], b, seed=7, sleep=False)
        self.assertEqual(len(a.spans), len(b.spans))

    def test_profile_fit_recovers_median(self):
        import random, math
        rng = random.Random(0)
        truth = LatencyProfile("x", median_s=2.0, sigma=0.4)
        samples = [truth.sample(rng) for _ in range(4000)]
        fit = LatencyProfile.from_samples("x", samples)
        self.assertAlmostEqual(fit.median_s, 2.0, delta=0.15)
        self.assertAlmostEqual(fit.sigma, 0.4, delta=0.05)

    def test_from_samples_rejects_too_few(self):
        with self.assertRaises(ValueError):
            LatencyProfile.from_samples("x", [1.0])


class TestAdapters(unittest.TestCase):
    def test_instrument_chat_records_span(self):
        t = Tracer()
        def fake(**kw):
            time.sleep(0.05)
            class U: prompt_tokens, completion_tokens = 10, 20
            class R: usage = U()
            return R()
        wrapped = instrument_chat(fake, t)
        with t.span("turn", SpanKind.TURN):
            wrapped(model="m")
        llm = [s for s in t.spans if s.kind is SpanKind.LLM_CALL][0]
        self.assertEqual(llm.attrs["model"], "m")
        self.assertEqual(llm.attrs["prompt_tokens"], 10)
        self.assertGreater(llm.blocked, 0.04)

    def test_instrument_tool_preserves_return(self):
        t = Tracer()
        wrapped = instrument_tool(lambda x: x * 2, t, name="double")
        with t.span("turn", SpanKind.TURN):
            self.assertEqual(wrapped(21), 42)


class TestLangGraphAdapter(unittest.TestCase):
    """No real LangGraph here (stdlib-only tests) — these fakes reproduce the
    one calling convention that matters: a node is `fn(state) -> partial_state`."""

    def test_instrument_node_steps_increment(self):
        t = Tracer()

        def node(state):
            return {"x": state["x"] + 1}

        wrapped = instrument_node(node, t, name="plan")
        with t.span("turn", SpanKind.TURN):
            wrapped({"x": 0})
            wrapped({"x": 1})
            wrapped({"x": 2})

        step_spans = [s for s in t.spans if s.kind is SpanKind.STEP]
        self.assertEqual([s.attrs["step"] for s in step_spans], [0, 1, 2])

        state_spans = [s for s in t.spans if s.kind is SpanKind.STATE_IO]
        self.assertEqual([s.attrs["step"] for s in state_spans], [0, 1, 2])
        self.assertEqual([s.name for s in state_spans], ["state@0", "state@1", "state@2"])

    def test_instrument_node_two_nodes_count_independently(self):
        t = Tracer()
        plan = instrument_node(lambda state: state, t, name="plan")
        act = instrument_node(lambda state: state, t, name="act")
        with t.span("turn", SpanKind.TURN):
            plan({})
            act({})
            plan({})

        plan_steps = [s.attrs["step"] for s in t.spans if s.name == "plan"]
        act_steps = [s.attrs["step"] for s in t.spans if s.name == "act"]
        self.assertEqual(plan_steps, [0, 1])
        self.assertEqual(act_steps, [0])

    def test_instrument_state_records_state_io_span(self):
        t = Tracer()

        def fake_put(config, checkpoint, metadata, new_versions):
            return {**config, "saved": True}

        wrapped = instrument_state(fake_put, t, name="checkpoint_put")
        with t.span("turn", SpanKind.TURN):
            out = wrapped({"thread": 1}, {}, {}, {})
        self.assertEqual(out, {"thread": 1, "saved": True})
        io = [s for s in t.spans if s.kind is SpanKind.STATE_IO]
        self.assertEqual(len(io), 1)
        self.assertEqual(io[0].name, "checkpoint_put")


class TestAggregate(unittest.TestCase):
    def test_aggregate_needs_runs(self):
        with self.assertRaises(ValueError):
            aggregate([])

    def test_aggregate_medians(self):
        runs = []
        for _ in range(5):
            t = Tracer()
            run_turn(DEFAULT_SHAPES["short_tool_loop"], t, seed=3, sleep=False)
            runs.append(summarise(t.spans))
        agg = aggregate(runs)
        self.assertEqual(agg["n_runs"], 5)
        self.assertEqual(agg["steps_p50"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
