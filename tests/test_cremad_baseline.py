import unittest

import torch

from AV_v1.models.baseline import AVBaseline, AudioBaseline, VisualBaseline


class CremadBaselineTest(unittest.TestCase):
    def test_av_baseline_returns_cremad_logits(self):
        model = AVBaseline(num_classes=6)
        model.eval()

        audio = torch.randn(2, 1, 64, 80)
        visual = torch.randn(2, 3, 2, 64, 64)

        with torch.no_grad():
            logits = model(audio, visual)
            diagnostics = model.forward_with_modal_logits(audio, visual)

        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(diagnostics["logits"].shape, (2, 6))
        self.assertEqual(diagnostics["audio_logits"].shape, (2, 6))
        self.assertEqual(diagnostics["visual_logits"].shape, (2, 6))
        self.assertTrue(torch.allclose(diagnostics["logits"], logits, atol=1e-5))
        self.assertEqual(model.audio_net.conv1.in_channels, 1)
        self.assertEqual(model.visual_net.conv1.in_channels, 3)

    def test_audio_baseline_returns_cremad_logits(self):
        model = AudioBaseline(num_classes=6)
        model.eval()

        audio = torch.randn(2, 1, 64, 80)

        with torch.no_grad():
            logits = model(audio)

        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(model.encoder.conv1.in_channels, 1)

    def test_visual_baseline_returns_cremad_logits(self):
        model = VisualBaseline(num_classes=6)
        model.eval()

        visual = torch.randn(2, 3, 2, 64, 64)

        with torch.no_grad():
            logits = model(visual)

        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(model.encoder.conv1.in_channels, 3)


if __name__ == "__main__":
    unittest.main()
