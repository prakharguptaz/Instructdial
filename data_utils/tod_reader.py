import argparse
import logging
from data_utils.data_reader import Dataset

from .tod_helpers.utils_general import *
from .tod_helpers.utils_multiwoz import *
from .tod_helpers.utils_camrest676 import *
from .tod_helpers.utils_woz import *
from .tod_helpers.utils_smd import *
from .tod_helpers.utils_frames import *
from .tod_helpers.utils_msre2e import *
from .tod_helpers.utils_taskmaster import *
from .tod_helpers.utils_metalwoz import *
from .tod_helpers.utils_schema import *

from os.path import join, dirname

logger = logging.getLogger(__name__)


class TodDataset(Dataset):
    def __init__(self, data_path, dataset, split='train'):
        self.data_path = data_path
        self.idx = 0
        self.examples, meta_data = load_tod(data_path, dataset, split)

        if 'slot_classes' in meta_data:
            self.slot_classes = meta_data['slot_classes']

        if 'db' in meta_data:
            self.db = meta_data['db']

        if 'act_classes' in meta_data:
            self.act_classes = meta_data['act_classes']


class args:
    def __init__(self, data_path):
        self.data_path = join(dirname(dirname(os.path.abspath(__file__))), data_path)
        self.holdout_dataset = '["multiwoz"]'
        self.example_type = 'turn'
        self.max_line = None
        self.ontology_version = ''
        self.domain_act = False
        self.only_last_turn = False
        self.task_name = ''
        self.dataset = '["multiwoz", "camrest676", "woz", "smd", "frames", "msre2e", "taskmaster", "metalwoz", "schema"]'


def load_tod(data_path, ds_name, split):
    args_dict = {'data_path': join(dirname(dirname(os.path.abspath(__file__))), data_path),
                 'holdout_dataset': '["multiwoz"]',
                 'example_type': 'turn',
                 'max_line': None,
                 'ontology_version': '',
                 'domain_act': False,
                 'only_last_turn': False,
                 'task_name': '',
                 'dataset': '["multiwoz", "camrest676", "woz", "smd", "frames", "msre2e", "taskmaster", "metalwoz", "schema"]'}

    ## Read datasets and create global set of candidate responses
    datasets = {}
    cand_uttr_sys = set()

    data_trn, data_dev, data_tst, data_meta = globals()["prepare_data_{}".format(ds_name)](args_dict)
    # held-out mwoz for now
    if ds_name in ast.literal_eval(args_dict['holdout_dataset']):
        datasets[ds_name] = {"train": data_trn, "dev": data_dev, "test": data_tst, "meta": data_meta}
    else:
        datasets[ds_name] = {"train": data_trn + data_dev + data_tst, "dev": [], "test": [], "meta": data_meta}

    return datasets[ds_name][split], data_meta
