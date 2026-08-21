import unittest

from bilcs.adapters import AcademicExplorerAdapter, RecommendationAdapter


class AdapterTests(unittest.TestCase):
    def test_recommendation_adapter_builds_generic_graph(self) -> None:
        graph = RecommendationAdapter().to_graph([{"user_id": "u", "item_id": "i"}], {"u": [1]})
        self.assertEqual(graph.u_ids, ("u",))

    def test_academic_adapter_builds_generic_graph(self) -> None:
        graph = AcademicExplorerAdapter().to_graph([{"researcher_id": "r", "paper_id": "p"}], {"r": [1, 0]})
        self.assertEqual(graph.v_ids, ("p",))
