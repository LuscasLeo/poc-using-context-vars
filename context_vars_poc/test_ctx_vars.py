import re

import pytest


class UserName(str):
    USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __new__(cls, value: str) -> "UserName":
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
