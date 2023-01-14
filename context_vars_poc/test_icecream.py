import contextlib
from dataclasses import dataclass
from contextvars import ContextVar
from typing import Optional

import pytest


@dataclass
class Icecream:
    flavor: str


icecream_context = ContextVar[Optional[Icecream]]("icecream_context", default=None)


def get_current_icecream() -> Icecream:
    icecream = icecream_context.get()
    if icecream is None:
        raise RuntimeError("No icecream in context")

    return icecream


@contextlib.contextmanager
def using_icecream(icecream: Icecream):
    token = icecream_context.set(icecream)
    try:
        yield
    finally:
        icecream_context.reset(token)


def create_icecream_text():
    icecream = get_current_icecream()
    return f"I'm eating {icecream.flavor} icecream"


def test_create_icecream_text():
    with using_icecream(Icecream("chocolate")):
        assert create_icecream_text() == "I'm eating chocolate icecream"


def test_create_icecream_text_without_context():
    with pytest.raises(RuntimeError) as excinfo:
        create_icecream_text()

    assert "No icecream in context" in str(excinfo.value)


def test_create_icecream_text_with_context():
    with using_icecream(Icecream("chocolate")):
        text_1 = create_icecream_text()
        with using_icecream(
            Icecream("vanilla")
        ):  ## Note that the chocolate icecream is overwritten by the vanilla icecream until the context is reset
            text_2 = create_icecream_text()

        text_3 = create_icecream_text()

    assert text_1 == "I'm eating chocolate icecream"
    assert text_2 == "I'm eating vanilla icecream"
    assert text_3 == "I'm eating chocolate icecream"
