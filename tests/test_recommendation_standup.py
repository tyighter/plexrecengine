import unittest

from app.config import settings
from app.services.plex_index import PlexProfile
from app.services.recommendation import RecommendationEngine


class FakePlexItem:
    def __init__(self, profile: PlexProfile):
        self.ratingKey = profile.rating_key
        self.title = profile.title
        self.year = profile.year or 2024
        self.viewCount = 0
        self.isWatched = False


class FakePlexService:
    def __init__(self, profiles: dict[int, PlexProfile]):
        self._profiles = profiles

    def fetch_item(self, rating_key: int, extra=None):
        profile = self._profiles.get(rating_key)
        return FakePlexItem(profile) if profile else None

    def poster_url(self, item):
        return f"/posters/{getattr(item, 'ratingKey', 'unknown')}"

    def recently_watched_movies(self, *args, **kwargs):
        return []

    def recently_watched_shows(self, *args, **kwargs):
        return []


class FakeIndex:
    def __init__(self, profiles: dict[int, PlexProfile], related: list[PlexProfile]):
        self._profiles = profiles
        self._related = related

    def profile_for_item(self, item):
        return self._profiles.get(getattr(item, "ratingKey", None))

    def related_profiles(self, source_profile, limit=None):
        return list(self._related[:limit] if limit else self._related)

    def _plot_similarity_scores(self, source_profile):
        return {profile.rating_key: 0.0 for profile in self._related}


class RecommendationStandupFilteringTests(unittest.TestCase):
    def setUp(self):
        self.original_flag = settings.standup_only_matching
        settings.standup_only_matching = True

    def tearDown(self):
        settings.standup_only_matching = self.original_flag

    def _make_engine(self, related_profiles):
        profiles = {profile.rating_key: profile for profile in related_profiles}
        source = PlexProfile(
            rating_key=1,
            title="Source",
            media_type="movie",
            genres={"Comedy"},
            summary="source",
            is_standup=True,
        )
        profiles[source.rating_key] = source
        plex = FakePlexService(profiles)
        index = FakeIndex(profiles, related_profiles)
        return RecommendationEngine(plex, index), source

    def test_standup_source_filters_non_standup_candidates(self):
        standup_candidate = PlexProfile(
            rating_key=2,
            title="Standup Special",
            media_type="movie",
            genres={"Comedy"},
            summary="standup",
            is_standup=True,
        )
        non_standup_candidate = PlexProfile(
            rating_key=3,
            title="Non Standup",
            media_type="movie",
            genres={"Comedy"},
            summary="movie",
            is_standup=False,
        )
        engine, source = self._make_engine([standup_candidate, non_standup_candidate])
        source_item = FakePlexItem(source)

        recommendations = engine.top_recommendations_for_item(source_item, media_type="movie", count=5)
        self.assertTrue(all(rec.rating_key != non_standup_candidate.rating_key for rec in recommendations))
        self.assertTrue(any(rec.rating_key == standup_candidate.rating_key for rec in recommendations))

    def test_non_standup_source_excludes_standup_candidates(self):
        standup_candidate = PlexProfile(
            rating_key=5,
            title="Comic Set",
            media_type="movie",
            genres={"Comedy"},
            summary="standup",
            is_standup=True,
        )
        standard_candidate = PlexProfile(
            rating_key=6,
            title="Comedy Movie",
            media_type="movie",
            genres={"Comedy"},
            summary="movie",
            is_standup=False,
        )
        engine, source = self._make_engine([standup_candidate, standard_candidate])
        source.is_standup = False
        source_item = FakePlexItem(source)

        recommendations = engine.top_recommendations_for_item(source_item, media_type="movie", count=5)
        self.assertTrue(all(rec.rating_key != standup_candidate.rating_key for rec in recommendations))
        self.assertTrue(any(rec.rating_key == standard_candidate.rating_key for rec in recommendations))


if __name__ == "__main__":
    unittest.main()
