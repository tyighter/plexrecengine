import unittest

import numpy as np

from app.config import settings
from app.services.plex_index import PlexIndex, PlexProfile, profile_similarity


class PlotSimilarityScoresTests(unittest.TestCase):
    def test_cosine_scores_preserve_separation_for_small_pool(self):
        index = PlexIndex(None)
        index._profiles = {}
        index._summary_matrix = None

        def normalized(vector: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vector)
            return vector / norm if norm else vector

        source = PlexProfile(
            rating_key=1,
            title="Source", 
            media_type="movie",
            summary="source summary",
        )
        close_match = PlexProfile(
            rating_key=2,
            title="Close Match",
            media_type="movie",
            summary="close summary",
        )
        far_match = PlexProfile(
            rating_key=3,
            title="Far Match",
            media_type="movie",
            summary="far summary",
        )

        embeddings = {
            source.summary: normalized(np.array([1.0, 0.0])),
            close_match.summary: normalized(np.array([0.9, 0.3])),
            far_match.summary: normalized(np.array([-1.0, 0.0])),
        }

        index._profiles = {
            source.rating_key: source,
            close_match.rating_key: close_match,
            far_match.rating_key: far_match,
        }

        index._encode_summaries = lambda summaries: np.array([
            embeddings[summary] for summary in summaries
        ])

        index._rebuild_text_matrix()

        scores = index._plot_similarity_scores(source)

        self.assertGreater(scores[close_match.rating_key], scores[far_match.rating_key])
        expected_close_score = float(np.dot(embeddings[source.summary], embeddings[close_match.summary]))
        self.assertAlmostEqual(scores[close_match.rating_key], expected_close_score, places=6)
        self.assertAlmostEqual(scores[far_match.rating_key], -1.0, places=6)
        self.assertGreater(scores[close_match.rating_key] - scores[far_match.rating_key], 1.5)


class SimilarityFieldWeightingTests(unittest.TestCase):
    def setUp(self):
        self._original_weights = {
            "studio": settings.studio_weight,
            "collection": settings.collection_weight,
            "country": settings.country_weight,
            "cast": settings.cast_weight,
            "director": settings.director_weight,
            "writer": settings.writer_weight,
            "genre": settings.genre_weight,
            "keyword": settings.keyword_weight,
            "letterboxd_keyword": settings.letterboxd_keyword_weight,
            "year": settings.year_weight,
        }
        settings.studio_weight = 10.0
        settings.collection_weight = 8.0
        settings.country_weight = 5.0
        settings.cast_weight = 0.0
        settings.director_weight = 0.0
        settings.writer_weight = 0.0
        settings.genre_weight = 0.0
        settings.keyword_weight = 0.0
        settings.letterboxd_keyword_weight = 0.0
        settings.year_weight = 0.0

    def tearDown(self):
        settings.studio_weight = self._original_weights["studio"]
        settings.collection_weight = self._original_weights["collection"]
        settings.country_weight = self._original_weights["country"]
        settings.cast_weight = self._original_weights["cast"]
        settings.director_weight = self._original_weights["director"]
        settings.writer_weight = self._original_weights["writer"]
        settings.genre_weight = self._original_weights["genre"]
        settings.keyword_weight = self._original_weights["keyword"]
        settings.letterboxd_keyword_weight = self._original_weights["letterboxd_keyword"]
        settings.year_weight = self._original_weights["year"]

    def test_collection_overlap_increases_similarity(self):
        source = PlexProfile(
            rating_key=1,
            title="Source",
            media_type="movie",
            summary="source",
            collections={"Franchise"},
        )

        related = PlexProfile(
            rating_key=2,
            title="Related",
            media_type="movie",
            summary="related",
            collections={"Franchise"},
        )

        unrelated = PlexProfile(
            rating_key=3,
            title="Unrelated",
            media_type="movie",
            summary="unrelated",
            collections=set(),
        )

        related_score, _ = profile_similarity(source, related, plot_score=0.0)
        unrelated_score, _ = profile_similarity(source, unrelated, plot_score=0.0)

        self.assertGreater(related_score, unrelated_score)

    def test_studio_overlap_increases_similarity(self):
        source = PlexProfile(
            rating_key=10,
            title="Source",
            media_type="tv",
            summary="source",
            studios={"Network"},
            countries={"USA"},
        )

        same_studio = PlexProfile(
            rating_key=11,
            title="Same Studio",
            media_type="tv",
            summary="same",
            studios={"Network"},
            countries={"USA"},
        )

        different_studio = PlexProfile(
            rating_key=12,
            title="Different Studio",
            media_type="tv",
            summary="different",
            studios={"Other"},
            countries={"USA"},
        )

        same_score, _ = profile_similarity(source, same_studio, plot_score=0.0)
        different_score, _ = profile_similarity(source, different_studio, plot_score=0.0)

        self.assertGreater(same_score, different_score)

class QualityWeightingTests(unittest.TestCase):
    def setUp(self):
        self._original_quality_weight = settings.quality_weight
        settings.quality_weight = 10.0

    def tearDown(self):
        settings.quality_weight = self._original_quality_weight

    def test_higher_rated_candidates_rank_first(self):
        source = PlexProfile(
            rating_key=1,
            title="Source",
            media_type="movie",
            summary="source",
        )

        high_quality = PlexProfile(
            rating_key=2,
            title="High",
            media_type="movie",
            summary="high",
            tmdb_rating=90.0,
        )

        low_quality = PlexProfile(
            rating_key=3,
            title="Low",
            media_type="movie",
            summary="low",
            tmdb_rating=60.0,
        )

        high_score, high_breakdown = profile_similarity(source, high_quality, plot_score=0.0)
        low_score, low_breakdown = profile_similarity(source, low_quality, plot_score=0.0)

        self.assertGreater(high_breakdown.get("quality", 0.0), low_breakdown.get("quality", 0.0))
        self.assertGreater(high_score, low_score)


if __name__ == "__main__":
    unittest.main()
