import importlib
from procnet.data_processor.DuEE_fin_processor import DuEEfinProcessor
from procnet.data_preparer.DuEE_fin_preparer import DuEEfinPreparer
import logging
importlib.reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


def run():
    du_pro = DuEEfinProcessor()

    du_pre = DuEEfinPreparer(processor=du_pro)

    du_pre.generate_pseudo_Doc2EDAG_data()


if __name__ == '__main__':
    run()
