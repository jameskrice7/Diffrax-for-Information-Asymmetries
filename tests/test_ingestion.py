import sys
import types
from unittest import mock

import pandas as pd

from finax.data.ingestion import fetch_quandl, fetch_yahoo, stream_quotes


def test_fetch_yahoo(monkeypatch):
    csv = "Date,Close\n2020-01-01,1.0\n"
    fake_resp = mock.Mock(text=csv)
    fake_resp.raise_for_status = mock.Mock()
    monkeypatch.setattr("requests.get", lambda *a, **k: fake_resp)
    df = fetch_yahoo("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["Close"] == 1.0


def test_fetch_quandl(monkeypatch):
    payload = {
        "dataset": {
            "column_names": ["Date", "Value"],
            "data": [["2020-01-01", 2.0]],
        }
    }
    fake_resp = mock.Mock()
    fake_resp.json.return_value = payload
    fake_resp.raise_for_status = mock.Mock()
    monkeypatch.setattr("requests.get", lambda *a, **k: fake_resp)
    df = fetch_quandl("WIKI/AAPL", api_key="demo")
    assert df.iloc[0]["Value"] == 2.0


def test_stream_quotes_websocket(monkeypatch):
    messages = ["{\"p\":1}", "{\"p\":2}"]

    class FakeWS:
        def __init__(self):
            self.i = 0

        def recv(self):
            msg = messages[self.i]
            self.i += 1
            return msg

        def close(self):
            pass

    fake_module = types.SimpleNamespace(create_connection=lambda url: FakeWS())
    monkeypatch.setitem(sys.modules, "websocket", fake_module)

    gen = stream_quotes(ws_url="ws://example")
    assert next(gen)["p"] == 1
    assert next(gen)["p"] == 2
    gen.close()


def test_stream_quotes_kafka(monkeypatch):
    class FakeMsg:
        def __init__(self, value):
            self.value = value

    class FakeConsumer:
        def __iter__(self):
            return iter([FakeMsg(b"{\"p\":3}"), FakeMsg(b"{\"p\":4}")])

    fake_module = types.SimpleNamespace(
        KafkaConsumer=lambda topic, bootstrap_servers: FakeConsumer()
    )
    monkeypatch.setitem(sys.modules, "kafka", fake_module)

    gen = stream_quotes(kafka_servers=["localhost:9092"], kafka_topic="t")
    assert next(gen)["p"] == 3
    assert next(gen)["p"] == 4
