import unittest

from AV_v1.train_video import build_dataset_args


class TrainVideoTest(unittest.TestCase):
    def test_builds_cremad_args(self):
        args = build_dataset_args(
            [
                "--dataset",
                "cremad",
                "--epochs",
                "2",
                "--output-dir",
                "runs/video_cremad",
            ]
        )

        self.assertEqual(args.dataset, "cremad")
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.output_dir, "runs/video_cremad")

    def test_builds_ks_args(self):
        args = build_dataset_args(
            [
                "--dataset",
                "ks",
                "--use-video-frames",
                "3",
                "--output-dir",
                "runs/video_ks",
            ]
        )

        self.assertEqual(args.dataset, "ks")
        self.assertEqual(args.use_video_frames, 3)
        self.assertEqual(args.output_dir, "runs/video_ks")


if __name__ == "__main__":
    unittest.main()
