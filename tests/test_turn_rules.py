"""End-of-turn rules on synthetic fixtures (ported from OpenVideoFDB WS9, plus multi-party cases)."""

from __future__ import annotations

from eot import turn_rules as R
from eot.turn_rules import Segment, Speaker, Utterance, Word


def seg(a, b):
    return Segment(float(a), float(b))


def utt(a, b, text):
    words = text.split()
    step = (b - a) / len(words)
    return Utterance(float(a), float(b), text,
                     [Word(w, a + i * step, a + (i + 1) * step) for i, w in enumerate(words)])


def events(vad_a, utts_a, vad_b, utts_b, **kw):
    return R.find_silences(Speaker("a", vad_a, utts_a), [Speaker("b", vad_b, utts_b)], include_excluded=True, **kw)


def test_complete_when_partner_takes_floor():
    vad_a = [seg(0, 3), seg(9, 11)]
    utts_a = [utt(0, 3, "so that is what I think about it"), utt(9, 11, "yes exactly right")]
    vad_b = [seg(3.6, 8)]
    utts_b = [utt(3.6, 8, "I see what you mean and I agree")]
    first = next(e for e in events(vad_a, utts_a, vad_b, utts_b) if e.t == 3.0)
    assert first.label == R.COMPLETE
    assert first.b_onset_s == 0.6 and first.gap_s == 6.0 and first.b_speaker == "b"
    assert first.a_turn_words == 8 and first.a_text.startswith("so that")
    assert first.b_text.startswith("I see")
    assert first.endpoint_bool is True


def test_incomplete_when_speaker_resumes_first():
    vad_a = [seg(0, 3), seg(4.2, 7)]
    utts_a = [utt(0, 3, "and then we went to the"), utt(4.2, 7, "the market on the corner")]
    vad_b = [seg(3.5, 3.9), seg(8, 10)]
    utts_b = [utt(3.5, 3.9, "mhm"), utt(8, 10, "that sounds nice indeed")]
    first = next(e for e in events(vad_a, utts_a, vad_b, utts_b) if e.t == 3.0)
    assert first.label == R.INCOMPLETE
    assert first.gap_s == 1.2 and first.b_onset_s == 5.0
    assert first.endpoint_bool is False


def test_short_gap_is_not_a_silence():
    vad_a = [seg(0, 3), seg(3.1, 5), seg(6, 7)]
    utts_a = [utt(0, 5, "one two three four five six")]
    assert [e.t for e in events(vad_a, utts_a, [], [])] == [5.0, 7.0]


def test_collision_excluded():
    vad_a = [seg(0, 3), seg(3.9, 6)]
    utts_a = [utt(0, 3, "I was going to say that"), utt(3.9, 6, "no wait let me finish")]
    vad_b = [seg(3.5, 6)]
    utts_b = [utt(3.5, 6, "well I think that is wrong")]
    first = next(e for e in events(vad_a, utts_a, vad_b, utts_b) if e.t == 3.0)
    assert first.label is None and first.reason == "collision"


def test_no_floor_when_partner_already_talking():
    vad_a = [seg(2.0, 2.5), seg(6, 9)]
    utts_a = [utt(2.0, 2.5, "right"), utt(6, 9, "so what happened after that")]
    vad_b = [seg(0, 5)]
    utts_b = [utt(0, 5, "and the whole thing fell apart in the end")]
    first = next(e for e in events(vad_a, utts_a, vad_b, utts_b) if e.t == 2.5)
    assert first.label is None and first.reason == "no_floor"


def test_short_turn_excluded():
    vad_a = [seg(1.0, 1.4), seg(3, 6)]
    utts_a = [utt(1.0, 1.4, "yeah"), utt(3, 6, "I guess that is one way to see it")]
    first = next(e for e in events(vad_a, utts_a, [], []) if e.t == 1.4)
    assert first.reason == "short_turn"


def test_long_silence_excluded():
    vad_a = [seg(0, 3), seg(10, 12)]
    utts_a = [utt(0, 3, "let me think about this one"), utt(10, 12, "okay so no")]
    first = next(e for e in events(vad_a, utts_a, [], []) if e.t == 3.0)
    assert first.reason == "long_silence"


def test_late_partner_is_long_silence_not_complete():
    vad_a = [seg(0, 3)]
    utts_a = [utt(0, 3, "what do you think about that")]
    vad_b = [seg(7, 9)]
    utts_b = [utt(7, 9, "I am not sure to be honest")]
    assert events(vad_a, utts_a, vad_b, utts_b)[0].reason == "long_silence"


def test_tail_excluded_and_final_region_can_be_complete():
    vad_a = [seg(0, 3)]
    utts_a = [utt(0, 3, "and that is the end of my story")]
    assert events(vad_a, utts_a, [], [])[0].reason == "tail"
    vad_b = [seg(3.4, 6)]
    utts_b = [utt(3.4, 6, "great story thanks for sharing")]
    assert events(vad_a, utts_a, vad_b, utts_b)[0].label == R.COMPLETE


def test_find_all_tags_every_speaker_and_excludes_by_default():
    a = Speaker("a", [seg(0, 3), seg(9, 11)],
                [utt(0, 3, "so that is what I think about it"), utt(9, 11, "yes exactly right")])
    b = Speaker("b", [seg(3.6, 8)], [utt(3.6, 8, "I see what you mean and I agree")])
    evs = R.find_all([a, b])
    assert all(e.label is not None for e in evs)
    assert {e.speaker for e in evs} == {"a", "b"}
    b_end = next(e for e in evs if e.speaker == "b")
    assert b_end.t == 8.0 and b_end.label == R.COMPLETE
    counts = R.summarize(R.find_all([a, b], include_excluded=True))
    assert counts[R.COMPLETE] == 2 and counts["tail"] == 1


# --- multi-party -----------------------------------------------------------------

def test_third_party_can_take_the_floor():
    a = Speaker("a", [seg(0, 3), seg(9, 11)],
                [utt(0, 3, "so that is what I think about it"), utt(9, 11, "yes exactly right")])
    b = Speaker("b", [seg(20, 22)], [utt(20, 22, "much later remark here")])
    c = Speaker("c", [seg(3.7, 8)], [utt(3.7, 8, "I disagree with that completely")])
    ev = next(e for e in R.find_silences(a, [b, c]) if e.t == 3.0)
    assert ev.label == R.COMPLETE and ev.b_speaker == "c" and ev.b_onset_s == 0.7


def test_third_party_floor_blocks_candidate():
    # c is talking as a's region ends: a was not holding the floor, whatever b did
    a = Speaker("a", [seg(2.0, 2.5), seg(6, 9)],
                [utt(2.0, 2.5, "right"), utt(6, 9, "so what happened after that")])
    b = Speaker("b", [], [])
    c = Speaker("c", [seg(0, 5)], [utt(0, 5, "and the whole thing fell apart in the end")])
    ev = next(e for e in R.find_silences(a, [b, c], include_excluded=True) if e.t == 2.5)
    assert ev.reason == "no_floor"


def test_backchannel_from_any_party_is_ignored():
    a = Speaker("a", [seg(0, 3), seg(4.2, 7)],
                [utt(0, 3, "and then we went to the"), utt(4.2, 7, "the market on the corner")])
    b = Speaker("b", [seg(3.3, 3.6)], [utt(3.3, 3.6, "mhm")])
    c = Speaker("c", [seg(3.5, 3.9)], [utt(3.5, 3.9, "yeah")])
    ev = next(e for e in R.find_silences(a, [b, c]) if e.t == 3.0)
    assert ev.label == R.INCOMPLETE and ev.gap_s == 1.2


def test_words_to_vad_merges_close_words():
    ws = [Word("a", 0.0, 0.3), Word("b", 0.35, 0.6), Word(",", 0.6, 0.6), Word("c", 1.0, 1.4)]
    regs = R.words_to_vad(ws)
    assert [(r.start, r.end) for r in regs] == [(0.0, 0.6), (1.0, 1.4)]
