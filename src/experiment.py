from __future__ import annotations
from pathlib import Path
from typing import Callable, Any
from datetime import datetime
import glob

from pydantic import TypeAdapter


class Experiment:
    def __init__(
        self, name: str, n_replications: int, variants: dict[str, Callable[[], Env]]
    ):
        self.name = name
        self.n_replications = n_replications
        self.variants = variants
        self.experiment_dir: Path | None = None

    def run(self, log_dir: Path):
        # confirm log_dir either does not exist or is a directory
        if Path.exists(log_dir) and not Path.is_dir(log_dir):
            raise ValueError(f"Expected log_dir `{log_dir}` to be a directory")

        # Make directory
        Path.mkdir(log_dir, parents=True, exist_ok=True)

        # Make directory for experiment
        experiment_dir = (
            Path(log_dir) / f"{self.name}_{datetime.now().strftime("%Y%m%d%H%M%S")}"
        )
        self.experiment_dir = experiment_dir
        Path.mkdir(experiment_dir)

        # make directory for variants
        for variant_name in self.variants:
            variant_dir = experiment_dir / variant_name
            Path.mkdir(variant_dir)

        # run experiment
        for trial_num in range(self.n_replications):
            for variant_name, variant_env_producer in self.variants.items():
                # make directory for trial
                num_digits = len(str(self.n_replications))
                trial_dir = (
                    experiment_dir
                    / variant_name
                    / f"trial_{str(trial_num).zfill(num_digits)}"
                )
                Path.mkdir(trial_dir)

                print(f"Running trial {trial_num} for variant {variant_name}")
                env = variant_env_producer()

                filepath = trial_dir / "start_state.json"
                with open(filepath, "w") as f:
                    f.write(env.serialized_state())

                env.run()

                filepath = trial_dir / "end_state.json"
                with open(filepath, "w") as f:
                    f.write(env.serialized_state())

                filepath = trial_dir / "event_log.json"
                with open(filepath, "w") as f:
                    f.write(env.serialize_log())

    def experiment_results(
        self,
        trial_metrics_calculator: Callable[
            [EnvState, list[EventUnion]], dict[str, Any]
        ],
    ):
        if not self.experiment_dir:
            raise Exception(
                "No value for `experiment_dir`. Confirm that experiment has been run."
            )

        experiment_data = {
            "experiment_name": self.name,
            "variants": {variant: [] for variant in self.variants},
        }

        for variant_name in experiment_data["variants"]:
            variant_path = self.experiment_dir / variant_name
            trial_paths = glob.glob(str(variant_path) + "/trial_*")

            for trial_path in trial_paths:
                trial = int(trial_path.split("_")[-1])

                # get state
                with open(Path(trial_path) / "start_state.json", "r") as f:
                    json_data = f.read()
                env_state = EnvState.model_validate_json(json_data)

                # get events
                with open(Path(trial_path) / "event_log.json", "r") as f:
                    json_data = f.read()
                event_log = TypeAdapter(list[EventUnion]).validate_json(json_data)

                trial_data = {
                    "trial_num": trial,
                    "metrics": trial_metrics_calculator(env_state, event_log),
                }
                experiment_data["variants"][variant_name].append(trial_data)
        return experiment_data


from .env import Env
from .state import EnvState
from .event import EventUnion
