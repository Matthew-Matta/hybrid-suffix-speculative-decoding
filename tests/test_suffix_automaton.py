from src.suffix_automaton import SuffixAutomaton


def test_basic_query():
    sa = SuffixAutomaton()
    sa.build([1, 2, 3, 1, 2])
    drafts, match = sa.query([1, 2], max_draft_len=4)
    assert match == 2 and drafts[0] == 3


def test_all_substrings_accepted():
    import random
    random.seed(0)
    s = [random.randint(0, 3) for _ in range(60)]
    sa = SuffixAutomaton()
    sa.build(s)
    subs = {tuple(s[i:j]) for i in range(len(s)) for j in range(i + 1, min(len(s), i + 8) + 1)}
    for sub in subs:
        st = 0
        for t in sub:
            assert t in sa.states[st].next, sub
            st = sa.states[st].next[t]
    assert sa.num_states() <= 2 * len(s)          # <= 2n-1 states (+ root)


def test_incremental_equals_bulk():
    s = [3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 2]
    a, b = SuffixAutomaton(), SuffixAutomaton()
    a.build(s)
    for t in s:
        b.extend_one(t)
    assert a.num_states() == b.num_states()
