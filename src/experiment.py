from __future__ import annotations
from pathlib import Path
from typing import Callable
from datetime import datetime


class Experiment:
    def __init__(
        self, name: str, n_replications: int, variants: dict[str, Callable[[], Env]]
    ):
        self.name = name
        self.n_replications = n_replications
        self.variants = variants

    def run(self, log_dir: Path):
        # check if log_dir already exists and is a directory
        if Path.exists(log_dir) and not Path.is_dir(log_dir):
            raise ValueError(f"Expected log_dir `{log_dir}` to be a directory")

        # Make directory
        Path.mkdir(log_dir, parents=True, exist_ok=True)

        # Make directory for experiment
        experiment_dir = (
            Path(log_dir) / f"{self.name}_{datetime.now().strftime("%Y%m%d%H%M%S")}"
        )
        Path.mkdir(experiment_dir)

        # make directory for variants
        for variant in self.variants:
            variant_dir = experiment_dir / variant
            Path.mkdir(variant_dir)

        # run experiment
        for trial_num in range(self.n_replications):
            for variant_name, variant_callable in self.variants.items():
                print(f"Running trial {trial_num} for variant {variant_name}")
                env = variant_callable()
                env.run()

                num_digits = len(str(self.n_replications))
                filepath = (
                    experiment_dir
                    / variant_name
                    / f"env_state_{str(trial_num).zfill(num_digits)}.json"
                )
                with open(filepath, "w") as f:
                    f.write(env.serialized_state())

                filepath = (
                    experiment_dir
                    / variant_name
                    / f"event_log_{str(trial_num).zfill(num_digits)}.json"
                )
                with open(filepath, "w") as f:
                    f.write(env.serialize_log())


from .env import Env
