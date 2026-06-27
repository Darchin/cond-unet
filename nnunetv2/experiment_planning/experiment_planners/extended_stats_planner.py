import numpy as np
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner


class ExtendedStatsExperimentPlanner(ExperimentPlanner):

    PERCENTILES = [10, 25, 33, 50, 66, 75, 90]

    def plan_experiment(self):
        # Parent builds the plan AND writes it to disk, then returns the dict
        plans = super().plan_experiment()

        transpose_forward = plans['transpose_forward']

        spacings = np.array(
            self.dataset_fingerprint['spacings']
        )[:, transpose_forward]

        shapes = np.array(
            self.dataset_fingerprint['shapes_after_crop']
        )[:, transpose_forward]

        for tag, arr in (('spacing', spacings), ('shape', shapes)):
            plans[f'original_min_{tag}_after_transp']  = arr.min(axis=0).tolist()
            plans[f'original_max_{tag}_after_transp']  = arr.max(axis=0).tolist()
            plans[f'original_mean_{tag}_after_transp'] = arr.mean(axis=0).tolist()
            plans[f'original_std_{tag}_after_transp']  = arr.std(axis=0).tolist()

            for p in self.PERCENTILES:
                plans[f'original_percentile_{p:02d}_{tag}_after_transp'] = (
                    np.percentile(arr, p, axis=0).tolist()
                )

        # Overwrite the file now that we've added our keys
        self.save_plans(plans)
        return plans