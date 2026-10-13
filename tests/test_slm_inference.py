
import unittest

from semantic.slm_inference import predict


class TestSLMInference(unittest.TestCase):

    def test_legitimate_message(self):
        result = predict(
            "Your delivery is delayed. Check tracking in the official app."
        )

        self.assertEqual(result["label"], "legitimate")
        self.assertEqual(result["class"], 0)

    def test_coercive_message(self):
        result = predict(
            "Pay the unexpected processing fee to my personal account "
            "immediately or your refund will be cancelled."
        )

        self.assertEqual(result["label"], "scam_coercion")
        self.assertEqual(result["class"], 1)

    def test_probability_values_are_valid(self):
        result = predict(
            "Your appointment is confirmed for tomorrow morning."
        )

        probabilities = result["class_probabilities"]

        self.assertIn("0", probabilities)
        self.assertIn("1", probabilities)

        for probability in probabilities.values():
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

        self.assertAlmostEqual(
            sum(probabilities.values()), 1.0, places=5
        )

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            predict("   ")

    def test_model_version_is_v5(self):
        result = predict("Your appointment is confirmed.")
        self.assertEqual(result["model_version"], "V5")


if __name__ == "__main__":
    unittest.main()
