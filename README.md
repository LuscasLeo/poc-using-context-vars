# POC For blog article

## Using

### Requirements

- Python 3.7+
- poetry

### Install

```bash
poetry install
```

# [Using Python's contextvars API](https://dev.to/luscasleo/using-pythons-contextvars-api-1iec)


Sometimes we want to build a flexible code that fits in multiple contexts, but it's complexity is too high to accomplish. Sometimes we want a function to perform in a way given such configuration and maybe completely different in another scenarios. For that scenarios we usually use constants which are commonly configured by environment variables, or even sending extra parameters to the object (a `bool`, `enum`, or anything like) in order to say to the function if it needs to execute action x or y. Sometimes it gets a real problem. While your application grow, you may will need to send this extar information through many and many layers of your code.

## Example

Let's suppose we have a User class that representes the User Domain Model. The User model has restrictions for it's username and email atributes:
```
- Users username should not have any other character than alphanumeric ones and _
- Users username need to have at least 6 characters and not more than 20
- Users email should fit the pattern <name>@<provider>.<com|net|etc>
```

To fit it's domain requirements we build these domain models
```python
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

```

### Validation switcher

So, what if we want to validate the User Domain Model only in certain cases, for instance, when we receive a payload from http requests on rest controllers, but NOT when we load the user from a repository/storage (database, file, etc)?

For this we make the following change:

```python
import re

import pytest


class UserName(str):
    USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __new__(cls, value: str, validate: str = True) -> "UserName":
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

    def __new__(cls, value: str, validate: bool = True) -> "Email":
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
    assert Email("", validate=False) == ""


def test_not_validate_username() -> None:
    assert UserName("", validate=False) == ""
```

But this solution is not that profitable in a long term. As said before, the complexity will be higher as the application grows, and the validation pointer will eventually be injected in functions where it shouldn't be. To prevent this we can make use of python's native `contextvars` module.
It's important to say that `contextlib` module and `contextvars` module are not the same thing. They have different purposes but they're related one to another and they're super useful together!

### How does contextvars works?
Context Vars isolate some pointer to a value given it's context. You gonna use the same call and get different results depending your local context.

Professional debuggers know call stacks very well. It is the sequence of recursive calls your code make through the code. Which called function has is's own scope, which can define your local variables and get global and sometimes even the parent's scope variables.

![The call stack on vscode](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/qcvnvim1ujq7t75po7hy.png)

The trick is: instead of passing the value from the top the bottom of all the call stack, getting and passing it as a argument, params os config class. All you need is to set the context variable value and the most begging of the call stack.

![Context Scope illustrated](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/j7xfzeq2jphcok0nqrrs.png)

### First things first

You will need a **static** point of memory that references the context variable. For that we'll use the class `ContextVar` from `contextvars` module:

```python
from contextvars import ContextVar
validation_context: ContextVar[bool] = ContextVar("validation_context", default=True)

```

Note that my context variable is a `bool`. But you can choose type you want.
Note that i have a initial value defined in the `default` kwarg at my ContextVar instance. That means that is there is no local context var set, this default one will be used.

### Setting up the local context variable

The context variable setup need to follow three steps in order to prevent overloading the application memory with abandoned context vars that will not be used anymore: You define the local context variable, you process the code and then resets the local context variable.

```python
def using_validation_context():
    token: Token[bool] = validation_context.set(False)

    ## Call some function that after 4 unrelated calls will use validation value

    validation_context.reset(token)
```
and then when we need to use the validation information we just need to call the static context var pointer `get()`

```python
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
```

In order to improve the usability of context vars and prevent memory leak, we make use of `contextlib` context managers hook:
```python
@contextlib.contextmanager
def using_validation_context(validate: bool):
    token: Token[bool] = validation_context.set(validate)

    try:
        yield
    finally:
        validation_context.reset(token)
```

Now every time you need to use a value for a whole context you just need to use the `with` statement:

```python
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

```


## Extra

### Does a contextvar works across multiple threads?
The answer is NO. As you create and run a new thread, you are creating a whole new call stack. In order to use contextlib into this new thread, you need to make sure to setup the context var lib at the beginning of the call stack just like the main thread does, otherwise, the thread will get the default value from the static pointer.

### Strict context vars
Sometimes you don't want or cannot provide a initial state or default value to your context. And you need to make sure that in order to execute a function, you need to have a valid context variable value.

```python
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
```

### Real world example
In a real application we tend to write the more reliable code we can. For those cases we choose to follow some patterns, like, hexagonal architecture, clean arch, clean code, DDD and many others concepts.
In this case is exceptional to separate the domain model from the implementations. For instance, the User Domain Entity, from the User ORM entity that references a table on the database.

For instance, a database session that is used across multiple repositories in order to commit all the changes in a single commit, in another words, a [unit of work].(https://www.cosmicpython.com/book/chapter_06_uow.html)
I this example, the domain entity and the storage engine are segregated and create a isolated session for every time we call a use case.

![Unit of work in action](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/nkcngythfghu1g5jdp82.png)

For this case we have the following code:
```python
import contextlib
import re
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import TracebackType
from typing import Generator, Optional, Type

from sqlalchemy import Column, Integer, MetaData, String, Table
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import registry  # type: ignore
from sqlalchemy.orm.session import Session, sessionmaker

# region Unit Of Work Definition


class UnitOfWork(ABC):
    @staticmethod
    def _get_context_uow() -> "UnitOfWork":
        context = ctx_var_current_uow.get()
        if context is None:
            raise RuntimeError("No context session")
        return context

    __current_ctx_token: Optional["Token[Optional[UnitOfWork]]"] = None

    def set_current_context(self) -> None:
        if self.__current_ctx_token is not None:
            raise RuntimeError("Already have a current context token")

        token = ctx_var_current_uow.set(self)
        self.__current_ctx_token = token

    def remove_current_context(self) -> None:
        if self.__current_ctx_token is None:
            raise RuntimeError("No current context token")
        ctx_var_current_uow.reset(self.__current_ctx_token)

    @abstractmethod
    def commit(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def rollback(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __enter__(self) -> "UnitOfWork":
        return self

    @abstractmethod
    def __exit__(
        self, exc_type: Type[Exception], exc_val: Exception, exc_tb: TracebackType
    ) -> None:
        raise NotImplementedError()


ctx_var_current_uow = ContextVar[Optional[UnitOfWork]]("current_uow", default=None)


class UnitOfWorkFactory(ABC):
    def __call__(self) -> UnitOfWork:
        raise NotImplementedError()


# endregion


# region Unit of Work ORM Implementation with SQLAlchemy


class SQAAUnitOfWork(UnitOfWork):
    @staticmethod
    def get_sqa_current_uow() -> "SQAAUnitOfWork":
        context = UnitOfWork._get_context_uow()
        if not isinstance(context, SQAAUnitOfWork):
            raise RuntimeError("Context session is not a SQAAUnitOfWork")
        return context

    @staticmethod
    def get_current_session() -> Session:
        return SQAAUnitOfWork.get_sqa_current_uow().session

    @property
    def session(self) -> Session:
        if not self._session:
            raise RuntimeError("No session")
        return self._session

    _session: Session

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def __enter__(self) -> "SQAAUnitOfWork":
        self.set_current_context()
        self._session: Session = self._session_factory()
        return self

    def __exit__(
        self, exc_type: Type[Exception], exc_val: Exception, exc_tb: TracebackType
    ) -> None:
        self.remove_current_context()

        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()
        if exc_type is not None:
            raise exc_val.with_traceback(exc_tb)

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    def close(self) -> None:
        return self.session.close()


class SQAUOWFactory(UnitOfWorkFactory):
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def __call__(self) -> SQAAUnitOfWork:
        return SQAAUnitOfWork(session_factory=self.session_factory)


# endregion


# region Domain Models


validation_context: ContextVar[bool] = ContextVar("validation_context", default=True)


@contextlib.contextmanager
def using_validation_context(validate: bool) -> Generator[None, None, None]:
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


@dataclass
class User:
    username: UserName
    email: Email


# endregion


# region Domain Logic


class UserStorage(ABC):
    @abstractmethod
    def get_user(self, username: UserName) -> User:
        raise NotImplementedError()

    @abstractmethod
    def create_user(self, user: User) -> None:
        raise NotImplementedError()

    @abstractmethod
    def user_exists(self, username: UserName) -> bool:
        raise NotImplementedError()


class CreateUserUsecase:
    def __init__(
        self, user_storage: UserStorage, uow_factory: UnitOfWorkFactory
    ) -> None:
        self.user_storage = user_storage
        self.uow_factory = uow_factory

    def execute(self, user: User) -> None:
        with self.uow_factory() as uow:

            with using_validation_context(False):
                if self.user_storage.user_exists(user.username):
                    raise ValueError("User already exists")

                self.user_storage.create_user(user)
                uow.commit()


# endregion


# region Domain Models implementation with SQLAlchemy


mapper_registry = registry()  # type: ignore

metadata = MetaData()


## The implementation must OBEY the domain model interface
# For example, the username column has a length of 20, obeying the domain model
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(20), nullable=False, unique=True),
    Column("email", String(50), nullable=False, unique=True),
)

mapper_registry.map_imperatively(User, users_table)

# endregion

# region Domain Logic implementation with SQLAlchemy


class SQAAUserStorage(UserStorage):
    @staticmethod
    def get_current_session() -> Session:
        return SQAAUnitOfWork.get_current_session()

    def get_user(self, username: UserName) -> User:
        session = self.get_current_session()
        row = session.query(User).filter_by(username=username).one()
        return row

    def create_user(self, user: User) -> None:
        session = self.get_current_session()
        session.add(user)

    def user_exists(self, username: UserName) -> bool:
        session = self.get_current_session()
        return session.query(User).filter(User.username == username).count() > 0


# endregion

# region Application


def main() -> None:
    # setup
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    uow_factory = SQAUOWFactory(session_factory=session_factory)
    user_storage = SQAAUserStorage()

    # create instances of the domain logic injecting the dependencies
    usecase = CreateUserUsecase(user_storage=user_storage, uow_factory=uow_factory)

    # execute the usecase

    username = "Lusca"
    user_mail = "lusca@gg.com"

    user = User(username=UserName(username), email=Email(user_mail))
    usecase.execute(user)

    # check the result
    with uow_factory() as uow:
        user = user_storage.get_user(UserName(username))
        assert user.username == "Lusca"
        assert user.email == "lusca@gg.com"


# endregion

if __name__ == "__main__":
    main()

```
 You may say it's is a lot of code to do a simple thing. I agree with you totally. It's a trade-off. In exchange to have many classes and interfaces, you have a better testable application. 
You don't have make overusing of mocks and another tricks. You can make unit tests for every aspect of your domain models, domain logic or even your adapters. 
Your code is now modular, that is, if or when you need to change some adapter for another, because of infrastructure demand or for tests purposes, like, `InMemoryUsersStorage` it is simpler. 
Your domain is clear. You have almost a [Screaming Architecture](https://blog.cleancoder.com/uncle-bob/2011/09/30/Screaming-Architecture.html), that is, the application is saying to you what is it's purpose.

You can get get source code at https://github.com/LuscasLeo/poc-using-context-vars

## References
https://docs.python.org/3/library/contextvars.html


