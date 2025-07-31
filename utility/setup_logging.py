"""
Created on Tue May 27 18:12:39 2025
@author: jiazhen@emotiv.com
"""

import logging
import os

def setup_logging(result_dir: str, log_filename: str = "training_log.txt"):
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # === ✅ 避免重复添加 handler ===
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger, log_file


def data_size_logging(config, train_loader,val_loader,test_loader,logger):
    total_train = len(train_loader.dataset)
    total_val = len(val_loader.dataset) if val_loader is not None else 0
    total_test = len(test_loader.dataset) if test_loader is not None else 0
    total = total_train + total_val + total_test
    percent = lambda x: 100.0 * x / total if total > 0 else 0.0
    if getattr(config, "use_corpus", False):
        logger.info(f"Data: [Corpus]: Location at {config.corpus_data_path}")
    else:
        logger.info(f"Data: [{config.data_name}]")
    logger.info(f"Input shape: {next(iter(train_loader))[0].shape}")
    logger.info(f"Total number of trials: {total}")
    logger.info(f"  - Train: {total_train} ({percent(total_train):.2f}%)")
    logger.info(f"  - Validation: {total_val} ({percent(total_val):.2f}%)")
    logger.info(f"  - Test: {total_test} ({percent(total_test):.2f}%)")


class DummyLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass