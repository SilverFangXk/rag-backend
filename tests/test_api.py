from fastapi.testclient import TestClient

from app.config import get_settings
from app.llm import AnswerResult
from app.main import Services, create_app
from app.vector_store import Hit

HIT = Hit("experience--e1--0", "Experience", "Data Intern at Nexteer",
          "Experience: Data Intern at Nexteer\nDates: Jul 2026 - Present", 0.2)


class FakeRetriever:
    def __init__(self, hits): self.hits = hits
    def retrieve(self, q): return self.hits


class FakeLLM:
    def __init__(self, result): self.result, self.calls = result, 0
    def answer(self, q, hits):
        self.calls += 1
        return self.result


def make(hits=(HIT,), result=None, settings=None):
    llm = FakeLLM(result or AnswerResult(found=True, answer="She interns at Nexteer.", source_ids=[1]))
    svc = Services(settings or get_settings(), None, None, FakeRetriever(list(hits)), llm)
    return TestClient(create_app(svc)), llm


def test_grounded_answer_returns_cited_sources():
    c, _ = make()
    r = c.post("/api/chat", json={"question": "Where does she work?"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["answer"].startswith("She interns")
    assert body["sources"][0]["section"] == "Experience · Data Intern at Nexteer"


def test_no_relevant_chunks_skips_the_llm():
    c, llm = make(hits=())
    body = c.post("/api/chat", json={"question": "Capital of France?"}).json()
    assert body["found"] is False and body["sources"] == []
    assert llm.calls == 0


def test_llm_can_say_not_found():
    c, _ = make(result=AnswerResult(found=False, answer="That is not in her portfolio.", source_ids=[]))
    body = c.post("/api/chat", json={"question": "What is her salary?"}).json()
    assert body["found"] is False and body["sources"] == []


def test_smalltalk_does_not_hit_retrieval():
    c, llm = make(hits=())
    assert c.post("/api/chat", json={"question": "hello!"}).json()["found"] is True
    assert llm.calls == 0


def test_validation():
    c, _ = make()
    assert c.post("/api/chat", json={"question": "   "}).status_code == 422
    assert c.post("/api/chat", json={"question": "x" * 501}).status_code == 422
    assert c.post("/api/chat", json={}).status_code == 422


def test_cors_allows_only_configured_origin():
    c, _ = make()
    ok = c.options("/api/chat", headers={"Origin": "https://me.vercel.app",
                                          "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "content-type"})
    assert ok.headers.get("access-control-allow-origin") == "https://me.vercel.app"
    bad = c.options("/api/chat", headers={"Origin": "https://evil.example",
                                           "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_rate_limit():
    s = get_settings().model_copy(update={"rate_limit_per_minute": 3})
    c, _ = make(settings=s)
    codes = [c.post("/api/chat", json={"question": "Her skills?"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_pipeline_error_becomes_503():
    class Boom:
        def retrieve(self, q): raise RuntimeError("firestore down")
    svc = Services(get_settings(), None, None, Boom(), None)
    r = TestClient(create_app(svc)).post("/api/chat", json={"question": "Her skills?"})
    assert r.status_code == 503


def test_health():
    c, _ = make()
    assert c.get("/health").json()["status"] == "ok"


def test_client_ip_is_taken_from_the_right_not_the_forgeable_left():
    from types import SimpleNamespace

    from app.main import _client_ip

    def req(xff):
        return SimpleNamespace(headers={"x-forwarded-for": xff} if xff else {}, client=SimpleNamespace(host="9.9.9.9"))

    assert _client_ip(req("1.1.1.1, 2.2.2.2, 3.3.3.3")) == "3.3.3.3"      # last entry = added by the platform
    assert _client_ip(req("1.1.1.1, 2.2.2.2, 3.3.3.3"), hops=2) == "2.2.2.2"
    assert _client_ip(req("5.5.5.5")) == "5.5.5.5"
    assert _client_ip(req("")) == "9.9.9.9"


def test_spoofed_header_does_not_bypass_rate_limit():
    s = get_settings().model_copy(update={"rate_limit_per_minute": 2})
    c, _ = make(settings=s)
    codes = [c.post("/api/chat", json={"question": "Her skills?"},
                    headers={"X-Forwarded-For": f"10.0.0.{i}, 7.7.7.7"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]  # rotating the forged left entry changes nothing
