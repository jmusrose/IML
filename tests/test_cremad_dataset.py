import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from cremav1.datasets.cremad import CREMADVisualDataset, select_frame_indices


class CremadDatasetTest(unittest.TestCase):
    def test_select_frame_indices_ignores_unsorted_listdir_order_and_skips_first_frame(self):
        rng = np.random.default_rng(0)
        names = ["2.jpg", "0.jpg", "1.jpg"]

        indices = select_frame_indices(names, fps=1, rng=rng)

        self.assertIn(indices.tolist(), ([1], [2]))

    def test_select_frame_indices_sorts_selected_indices(self):
        rng = np.random.default_rng(1)
        names = ["frame_2.jpg", "frame_0.jpg", "frame_1.jpg", "frame_3.jpg"]

        indices = select_frame_indices(names, fps=2, rng=rng)

        self.assertEqual(indices.tolist(), sorted(indices.tolist()))
        self.assertTrue(all(index > 0 for index in indices))

    def test_select_frame_indices_uses_first_frame_when_it_is_the_only_option(self):
        rng = np.random.default_rng(0)

        indices = select_frame_indices(["0.jpg"], fps=1, rng=rng)

        self.assertEqual(indices.tolist(), [0])

    def test_visual_dataset_returns_channels_before_time_by_default(self):
        with TemporaryDirectory() as tmpdir:
            sample_dir = Path(tmpdir) / "Image-01-FPS" / "1001_IEO_HAP_LO"
            sample_dir.mkdir(parents=True)
            for index in range(3):
                image = Image.new("RGB", (8, 8), color=(index * 40, 0, 0))
                image.save(sample_dir / f"{index}.jpg")

            dataset = CREMADVisualDataset(tmpdir, fps=2, rng=np.random.default_rng(0))

            images, label = dataset[0]

            self.assertEqual(tuple(images.shape), (3, 2, 8, 8))
            self.assertEqual(label.item(), 3)


if __name__ == "__main__":
    unittest.main()
