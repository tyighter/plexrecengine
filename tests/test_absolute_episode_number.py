import sys
from types import ModuleType

plexapi_stub = ModuleType("plexapi")
sys.modules.setdefault("plexapi", plexapi_stub)

exceptions_stub = ModuleType("plexapi.exceptions")
exceptions_stub.NotFound = Exception
sys.modules.setdefault("plexapi.exceptions", exceptions_stub)

server_stub = ModuleType("plexapi.server")


class _FakePlexServer:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("PlexServer should not be constructed in tests")


server_stub.PlexServer = _FakePlexServer
sys.modules.setdefault("plexapi.server", server_stub)

pydantic_stub = ModuleType("pydantic")


class _FakeHttpUrl(str):
    pass


def _identity_decorator(*_args, **_kwargs):
    def decorator(fn):
        return fn

    return decorator


def _computed_field(fn=None, **_kwargs):
    if fn is None:
        return lambda inner: inner
    return property(fn)


pydantic_stub.HttpUrl = _FakeHttpUrl
pydantic_stub.computed_field = _computed_field
pydantic_stub.field_validator = _identity_decorator
pydantic_stub.model_validator = _identity_decorator
sys.modules.setdefault("pydantic", pydantic_stub)

pydantic_settings_stub = ModuleType("pydantic_settings")


class _FakeBaseSettings:
    def __init__(self, **_kwargs):
        pass


class _FakeSettingsConfigDict(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)


pydantic_settings_stub.BaseSettings = _FakeBaseSettings
pydantic_settings_stub.SettingsConfigDict = _FakeSettingsConfigDict
sys.modules.setdefault("pydantic_settings", pydantic_settings_stub)

yaml_stub = ModuleType("yaml")
yaml_stub.safe_load = lambda *_args, **_kwargs: {}
yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
sys.modules.setdefault("yaml", yaml_stub)

httpx_stub = ModuleType("httpx")
httpx_stub.get = lambda *_args, **_kwargs: (_ for _ in ()).throw(
    RuntimeError("httpx.get should not be invoked in tests")
)
sys.modules.setdefault("httpx", httpx_stub)

from app.services.plex_service import PlexService


class _FakeShow:
    def __init__(self, episodes):
        self._episodes = episodes

    def episodes(self):
        return list(self._episodes)


class _FakeEpisode:
    def __init__(
        self,
        rating_key: int,
        season: int,
        number: int,
        absolute_index=None,
        show=None,
    ):
        self.ratingKey = rating_key
        self.parentIndex = season
        self.index = number
        self.absoluteIndex = absolute_index
        self._show = show

    def show(self):
        return self._show


def _service() -> PlexService:
    return PlexService.__new__(PlexService)


def test_returns_existing_absolute_index():
    target = _FakeEpisode(rating_key=10, season=1, number=3, absolute_index=5)

    number, ordinal = _service().absolute_episode_number(target)

    assert number == 5
    assert ordinal == "5th"


def test_infers_absolute_index_from_show_episodes():
    target = _FakeEpisode(rating_key=20, season=2, number=1)
    show = _FakeShow(
        [
            _FakeEpisode(rating_key=11, season=1, number=1),
            _FakeEpisode(rating_key=12, season=1, number=2),
            target,
            _FakeEpisode(rating_key=21, season=2, number=2),
        ]
    )
    target._show = show

    number, ordinal = _service().absolute_episode_number(target)

    assert number == 3
    assert ordinal == "3rd"


def test_defaults_to_position_when_episode_indexes_missing():
    target = _FakeEpisode(rating_key=None, season=None, number=None)
    episodes = [
        _FakeEpisode(rating_key=101, season=1, number=1),
        _FakeEpisode(rating_key=102, season=1, number=2),
        target,
        _FakeEpisode(rating_key=201, season=2, number=1),
    ]
    show = _FakeShow(episodes)
    target._show = show

    number, ordinal = _service().absolute_episode_number(target)

    assert number == 3
    assert ordinal == "3rd"


def test_falls_back_to_episode_index_when_no_show_loaded():
    target = _FakeEpisode(rating_key=11, season=1, number=4)
    target._show = None

    number, ordinal = _service().absolute_episode_number(target)

    assert number == 4
    assert ordinal == "4th"

