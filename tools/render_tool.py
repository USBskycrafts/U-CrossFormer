import os
import time
import torch
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
from typing import Dict, List
from matplotlib import pyplot as plt


class ResultRender:
    def __init__(self, config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.lock = Lock()
        self.render_path = os.path.join(
            config.get("test", "result_path"),
            config.get("output", "model_name"),
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
        self.futures = []

    def __del__(self):
        wait(self.futures)

    def render(self, data: Dict[str, List[torch.Tensor]]):
        self.executor.submit(self.task, data)

    def task(self, data):
        with self.lock:
            # TODO: render
            modals, data_lists = data.keys(), data.values()
            for i, title, data in enumerate(zip(modals, data_lists)):
                ax = plt.subplot(1, len(modals), i + 1)
