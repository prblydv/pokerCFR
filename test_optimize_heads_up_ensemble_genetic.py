import unittest

from optimize_heads_up_ensemble_genetic import (
    FitnessSettings,
    candidate_id,
    evolve_population,
    initial_population,
    score_matches,
)


def match(ev, all_in_ev, non_all_in_ev, *, hands=20_000, stderr=0.25):
    return {
        "mean_ev_bb_per_100": float(ev),
        "paired_stderr_bb_per_hand": float(stderr) / 100.0,
        "candidate_all_in_net_bb": float(all_in_ev) * hands / 100.0,
        "candidate_non_all_in_net_bb": float(non_all_in_ev) * hands / 100.0,
        "hands": int(hands),
    }


class GeneticEnsembleFitnessTests(unittest.TestCase):
    def test_decomposition_sums_to_raw_ev(self):
        result = score_matches([match(7.0, 4.0, 3.0)], FitnessSettings())
        self.assertAlmostEqual(result["average_ev_bb_per_100"], 7.0)
        self.assertAlmostEqual(
            result["all_in_ev_bb_per_100"]
            + result["non_all_in_ev_bb_per_100"],
            7.0,
        )

    def test_balanced_profit_beats_equal_ev_all_in_dependency(self):
        settings = FitnessSettings()
        balanced = score_matches([match(10.0, 5.0, 5.0)], settings)
        dependent = score_matches([match(10.0, 30.0, -20.0)], settings)
        self.assertGreater(balanced["fitness"], dependent["fitness"])
        self.assertGreater(dependent["non_all_in_loss_penalty"], 0.0)
        self.assertGreater(dependent["all_in_concentration_penalty"], 0.0)

    def test_worst_opponent_loss_is_visible_and_penalized(self):
        settings = FitnessSettings(worst_opponent_loss_penalty=0.5)
        result = score_matches(
            [match(12.0, 6.0, 6.0), match(-8.0, 2.0, -10.0)], settings
        )
        self.assertEqual(result["worst_opponent_ev_bb_per_100"], -8.0)
        self.assertAlmostEqual(result["worst_opponent_loss_penalty"], 4.0)


class GeneticPopulationTests(unittest.TestCase):
    def test_initial_and_evolved_populations_obey_simplex_and_active_limits(self):
        iterations = [725, 950, 1025, 200, 275, 300, 400]
        population = initial_population(
            iterations,
            population_size=10,
            min_active=2,
            max_active=5,
            min_weight=0.02,
            seed=17,
        )
        self.assertEqual(len(population), 10)
        ranked = []
        for index, weights in enumerate(population):
            self.assertAlmostEqual(sum(weights), 1.0)
            self.assertIn(sum(weight > 0.0 for weight in weights), range(2, 6))
            ranked.append({"weights": weights, "fitness": float(10 - index)})
        evolved = evolve_population(
            ranked,
            population_size=10,
            elite_count=3,
            min_active=2,
            max_active=5,
            min_weight=0.02,
            mutation_scale=0.35,
            structural_mutation_probability=0.30,
            random_immigrants=2,
            seed=19,
        )
        self.assertEqual(len(evolved), 10)
        self.assertEqual(len({candidate_id(weights) for weights in evolved}), 10)
        self.assertEqual(evolved[:3], population[:3])
        for weights in evolved:
            self.assertAlmostEqual(sum(weights), 1.0)
            self.assertIn(sum(weight > 0.0 for weight in weights), range(2, 6))


if __name__ == "__main__":
    unittest.main()
