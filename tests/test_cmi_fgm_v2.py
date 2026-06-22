import unittest

import torch


class CMIFGMV2Test(unittest.TestCase):
    def test_fgm_state_uses_previous_signal_for_coefficients(self):
        from cmi_fgm import CMIFGMState

        state = CMIFGMState(
            modalities=("audio", "visual"),
            strength=0.5,
            temperature=1.0,
            momentum=0.0,
            warmup_steps=0,
        )
        initial = state.coefficients(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

        state.update(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        coefs = state.coefficients(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

        self.assertTrue(torch.allclose(initial["audio"], torch.ones(2)))
        self.assertGreater(coefs["audio"][0].item(), coefs["visual"][0].item())
        self.assertGreater(coefs["visual"][1].item(), coefs["audio"][1].item())
        self.assertGreater(coefs["audio"][0].item(), 1.0)
        self.assertGreater(coefs["visual"][1].item(), 1.0)

    def test_av_v2_probe_heads_do_not_change_fusion_logits(self):
        from AV_v2.models import AVBaseline

        model = AVBaseline(num_classes=6)
        model.eval()
        audio = torch.randn(2, 1, 64, 80)
        visual = torch.randn(2, 3, 2, 64, 64)

        with torch.no_grad():
            before = model.forward_with_modal_logits(audio, visual)["logits"]
            for head in (model.audio_probe, model.visual_probe):
                for param in head.parameters():
                    param.add_(100.0)
            outputs = model.forward_with_modal_logits(audio, visual)

        self.assertEqual(tuple(outputs["logits"].shape), (2, 6))
        self.assertEqual(tuple(outputs["audio_logits"].shape), (2, 6))
        self.assertEqual(tuple(outputs["visual_logits"].shape), (2, 6))
        self.assertTrue(torch.allclose(outputs["logits"], before, atol=1e-5))

    def test_av_v2_training_reports_fgm_metrics_when_enabled(self):
        from AV_v2.models import AVBaseline
        from AV_v2.train_cremad import forward_and_losses
        from cmi_fgm import CMIFGMState

        model = AVBaseline(num_classes=6)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        fgm_state = CMIFGMState(("audio", "visual"), strength=0.5, warmup_steps=0, momentum=0.0)
        audio = torch.randn(2, 1, 64, 80)
        visual = torch.randn(2, 3, 2, 64, 64)
        labels = torch.tensor([0, 1], dtype=torch.long)

        _, first_losses, first_handles = forward_and_losses(
            model,
            (audio, visual),
            labels,
            "av",
            criterion,
            fgm_state=fgm_state,
        )
        first_losses["loss"].backward()
        for handle in first_handles:
            handle.remove()

        model.zero_grad(set_to_none=True)
        _, second_losses, second_handles = forward_and_losses(
            model,
            (audio, visual),
            labels,
            "av",
            criterion,
            fgm_state=fgm_state,
        )
        second_losses["loss"].backward()
        for handle in second_handles:
            handle.remove()

        self.assertIn("fgm_coef_audio", second_losses)
        self.assertIn("fgm_signal_audio", second_losses)
        self.assertGreaterEqual(second_losses["fgm_coef_audio"].item(), 1.0)
        self.assertGreaterEqual(second_losses["fgm_coef_visual"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
