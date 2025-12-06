import unittest

import numpy as np

from app.services.plex_index import PlexIndex, PlexProfile


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


if __name__ == "__main__":
    unittest.main()
