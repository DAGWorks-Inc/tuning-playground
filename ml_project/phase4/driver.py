from hamilton.driver import Builder, Driver
from hamilton.io.materialization import to, from_
import full_dag
import training__v1
import training__v2 #TODO: automatically add all modules?
import logging
from omegaconf import DictConfig, OmegaConf
import hydra

# Setup a logger
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    # Log the working and output directories
    logger.info("Starting application with configuration:\n%s", OmegaConf.to_yaml(cfg))

    cfg_dct = {"mode": cfg.mode, "model_type": cfg.model.model_type}

    if cfg.mode == "inference":
        mats = [
            from_.pickle(
                target="trained_model",
                path=cfg.model.model_path,
            )
        ]
    else:
        mats = [
            to.pickle(
                dependencies=["trained_model"],
                id="model_to_pickle",
                path=cfg.model.model_path,
            )
        ]

    if cfg.model.model_type == "v1":
        modules = [full_dag, training__v1]
    else:
        modules = [full_dag, training__v2]

    dr = (
        Builder()
        .with_modules(*modules)
        .with_config(cfg_dct)
        .with_materializers(*mats)
    ).build()

    data_path = "data/01_raw/dataset.parquet"

    inputs = {
        "data_path": data_path,
        "n_estimators": cfg.model.n_estimators,
    }

    logger.info("Starting application...")

    if cfg.mode == "inference":
        final_vars = ["predictions"]
    else:
        final_vars = ["accuracy", "model_to_pickle"]

    results = dr.execute(final_vars, inputs=inputs)
    logger.info(f"Results: {results}")
    logger.info("Dataflow execution completed.")

    dr.display_all_functions("dag.png")


if __name__ == "__main__":
    main()
