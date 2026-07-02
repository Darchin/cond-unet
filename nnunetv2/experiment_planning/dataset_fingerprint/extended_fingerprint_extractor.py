import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor


class ExtendedFingerprintExtractor(DatasetFingerprintExtractor):
    PERCENTILES = [10, 25, 33, 50, 66, 75, 90]

    def run(self, overwrite_existing: bool = False) -> dict:
        fingerprint = super().run(overwrite_existing=overwrite_existing)

        spacings = np.array(fingerprint['spacings'])
        shapes = np.array(fingerprint['shapes_after_crop'])

        for tag, arr in (('spacing', spacings), ('shape', shapes)):
            fingerprint[f'original_min_{tag}'] = arr.min(axis=0).tolist()
            fingerprint[f'original_max_{tag}'] = arr.max(axis=0).tolist()
            fingerprint[f'original_mean_{tag}'] = arr.mean(axis=0).tolist()
            fingerprint[f'original_std_{tag}'] = arr.std(axis=0).tolist()

            for p in self.PERCENTILES:
                fingerprint[f'original_percentile_{p:02d}_{tag}'] = (
                    np.percentile(arr, p, axis=0).tolist()
                )

        # Save updated fingerprint to file
        preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')
        save_json(fingerprint, properties_file)

        return fingerprint
