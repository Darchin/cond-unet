from typing import Dict, List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, save_json

from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from nnunetv2.paths import nnUNet_preprocessed


class CondUNetFingerprintExtractor(DatasetFingerprintExtractor):
    spacing_percentile_step = 5

    @staticmethod
    def compute_spacing_percentiles(spacings: List[List[float]], percentile_step: int = 5) -> Dict[str, List[float]]:
        spacings = np.vstack(spacings)
        return {
            str(p): [float(i) for i in np.percentile(spacings, p, axis=0)]
            for p in range(0, 101, percentile_step)
        }

    def run(self, overwrite_existing: bool = False) -> dict:
        fingerprint = super().run(overwrite_existing)
        if "spacing_percentiles" not in fingerprint:
            fingerprint["spacing_percentiles"] = self.compute_spacing_percentiles(
                fingerprint["spacings"], self.spacing_percentile_step
            )
            save_json(
                fingerprint,
                join(nnUNet_preprocessed, self.dataset_name, "dataset_fingerprint.json"),
                sort_keys=False,
            )
        return fingerprint
