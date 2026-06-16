"""
test_all.py - Full module integration test
Run: python test_all.py
"""
import os
import sys
os.environ["PYTHONUTF8"] = "1"

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(label, fn):
    try:
        msg = fn()
        print(f"{PASS} {label}" + (f" — {msg}" if msg else ""))
        results.append((label, True, ""))
    except Exception as e:
        print(f"{FAIL} {label} — {e}")
        results.append((label, False, str(e)))

# ── 1. utils.py ──────────────────────────────────────────────
def test_utils():
    from utils import TextPreprocessor
    tp = TextPreprocessor()
    result = tp.preprocess("What is the admission fee for college?")
    assert result, "Empty preprocessed output"
    return f"'{result}'"
check("utils.py — TextPreprocessor", test_utils)

# ── 2. faq_manager.py ────────────────────────────────────────
def test_faq_manager():
    from faq_manager import FAQManager
    mgr = FAQManager()
    assert mgr.count > 0, "No FAQs loaded"
    return f"{mgr.count} FAQs loaded"
check("faq_manager.py — FAQManager", test_faq_manager)

# ── 3. chatbot.py ────────────────────────────────────────────
def test_chatbot():
    from faq_manager import FAQManager
    from chatbot import FAQEngine
    mgr = FAQManager()
    engine = FAQEngine(faq_manager=mgr)
    
    tests = [
        ("How much is the college fee?",   True),
        ("admission fee",                  True),
        ("hostel accommodation cost",      True),
        ("xyzzy gibberish blah blah blah", False),  # should fail gracefully
    ]
    passed = 0
    for q, expect_success in tests:
        resp = engine.query(q)
        if resp.is_successful == expect_success:
            passed += 1
    return f"{passed}/{len(tests)} semantic queries matched expected outcome"
check("chatbot.py — FAQEngine + semantic search", test_chatbot)

# ── 4. pdf_processor.py ──────────────────────────────────────
def test_pdf_processor():
    from pdf_processor import PDFProcessor
    proc = PDFProcessor(chunk_size=300, chunk_overlap=50)
    assert proc.chunk_size == 300
    return "PDFProcessor initialized OK"
check("pdf_processor.py — PDFProcessor", test_pdf_processor)

# ── 5. voice_handler.py ──────────────────────────────────────
def test_voice():
    from voice_handler import VoiceHandler
    vh = VoiceHandler()
    status = vh.get_status()
    stt = "ON" if status["stt_available"] else "OFF"
    tts = "ON" if status["tts_available"] else "OFF"
    backend = status["stt_backend"]
    return f"STT={stt} (backend={backend})  TTS={tts}"
check("voice_handler.py — VoiceHandler", test_voice)

# ── 6. analytics.py ──────────────────────────────────────────
def test_analytics():
    from faq_manager import FAQManager
    from chatbot import FAQEngine
    from analytics import AnalyticsManager
    mgr = FAQManager()
    engine = FAQEngine(faq_manager=mgr)
    am = AnalyticsManager()
    resp = engine.query("What is the hostel fee?")
    am.log_chat_response(resp)
    stats = am.get_summary_stats()
    assert stats["total_queries"] >= 1
    return f"Logged OK, total_queries={stats['total_queries']}"
check("analytics.py — AnalyticsManager", test_analytics)

# ── 7. Plotly chart generation ───────────────────────────────
def test_charts():
    from analytics import AnalyticsManager
    am = AnalyticsManager()
    figs_generated = 0
    for method in [am.plot_query_volume, am.plot_success_rate, am.plot_confidence_distribution,
                   am.plot_top_questions, am.plot_category_breakdown, am.plot_source_breakdown]:
        fig = method()
        if fig is not None:
            figs_generated += 1
    return f"{figs_generated} Plotly charts generated"
check("analytics.py — Plotly charts", test_charts)

# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 55)
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

if failed == 0:
    print(f"  ALL {total} TESTS PASSED - Project is ready!")
else:
    print(f"  {passed}/{total} passed, {failed} FAILED:")
    for label, ok, err in results:
        if not ok:
            print(f"    x {label}: {err}")
print("=" * 55)
sys.exit(0 if failed == 0 else 1)
