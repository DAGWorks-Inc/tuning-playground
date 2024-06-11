from hamilton import driver
import full_dag
import os
import logging
from omegaconf import DictConfig, OmegaConf
import hydra

# Setup a logger
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    # Log the working and output directories
    logger.info("Starting application with configuration:\n%s", OmegaConf.to_yaml(cfg))

    dr = driver.Builder().with_modules(full_dag).build()

    data_path = "data/01_raw/dataset.parquet"
    inputs = {"data_path": data_path, "n_estimators": cfg.model.n_estimators}

    final_vars = ["accuracy"]
    logger.info("Starting application...")
    results = dr.execute(final_vars, inputs=inputs)

    logger.info(f'Accuracy: {results["accuracy"]}')
    logger.info("Dataflow execution completed.")

if __name__ == "__main__":
    main()
