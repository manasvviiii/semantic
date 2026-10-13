
import unittest

from semantic.classifier import build_classifier


class TestSemanticClassifier(unittest.TestCase):

    def setUp(self):
        self.model = build_classifier()
        self.texts = [
            "Your appointment is confirmed for tomorrow morning.",
            "Share your OTP immediately or your bank account will be blocked.",
            "Your bank statement is available in your banking application.",
            "Do not tell anyone. Transfer the money immediately.",
        ]
        self.labels = [0, 1, 0, 1]
        self.model.fit(self.texts, self.labels)

    def test_legitimate_message(self):
        prediction = self.model.predict([
            "Your appointment is confirmed for tomorrow morning."
        ])
        self.assertEqual(prediction[0], 0)

    def test_scam_message(self):
        prediction = self.model.predict([
            "Share your OTP immediately or your bank account will be blocked."
        ])
        self.assertEqual(prediction[0], 1)

    def test_prediction_labels_are_valid(self):
        predictions = self.model.predict(self.texts)
        self.assertTrue(set(predictions).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()