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
