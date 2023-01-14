import contextlib
import re
from contextvars import ContextVar, Token

import pytest

validation_context: ContextVar[bool] = ContextVar("validation_context", default=True)


@contextlib.contextmanager
def using_validation_context(validate: bool):
    token: Token[bool] = validation_context.set(validate)

    try:
        yield
    finally:
        validation_context.reset(token)


class UserName(str):
    USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __new__(cls, value: str) -> "UserName":
        validate = validation_context.get()
        if validate:
            if not value:
                raise ValueError("Username cannot be empty")

            if len(value) > 20:
                raise ValueError("Username is too long")

            if len(value) < 3:
                raise ValueError("Username is too short")

            if not cls.USERNAME_PATTERN.fullmatch(value):
                raise ValueError("Invalid username")

        return super().__new__(cls, value)


class Email(str):
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

    def __new__(cls, value: str) -> "Email":
        validate = validation_context.get()
        if validate:
            if not value:
                raise ValueError("Email cannot be empty")

            if not cls.EMAIL_PATTERN.fullmatch(value):
                raise ValueError("Invalid email")

        return super().__new__(cls, value)


class User:
    username: UserName
    email: Email


def test_valid_username() -> None:
    username = UserName("john")
    assert username == "john"


def test_valid_email() -> None:
    email = Email("luscasleo@gg.com")
    assert email == "luscasleo@gg.com"


def test_invalid_username() -> None:
    with pytest.raises(ValueError) as excinfo:
        username = UserName("jo")

    assert "Username is too short" in str(excinfo.value)


def test_invalid_email() -> None:
    with pytest.raises(ValueError) as excinfo:
        email = Email("luscasleo@gg")

    assert "Invalid email" in str(excinfo.value)


def test_valid_user() -> None:
    user = User()
    user.username = UserName("luscasleo")
    user.email = Email("ll@gg.com")

    assert user.username == "luscasleo"
    assert user.email == "ll@gg.com"


def test_not_validate_email() -> None:
    with using_validation_context(False):
        assert Email("") == ""


def test_not_validate_username() -> None:
    with using_validation_context(False):
        assert UserName("") == ""


def test_username_validation_on_multithreading():
    def run():
        UserName("")

    from concurrent.futures.thread import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        with using_validation_context(False):
            UserName("")
            future = executor.submit(run)

            future.exception() is not None
