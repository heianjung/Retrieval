import os
import logging
import torch

def init_logger(path: str):
	if not os.path.exists(path):
		os.makedirs(path)
	logger = logging.getLogger()
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
	debug_fh.setLevel(logging.DEBUG)

	info_fh = logging.FileHandler(os.path.join(path, "info.log"))
	info_fh.setLevel(logging.INFO)

	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)

	info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
	debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

	ch.setFormatter(info_formatter)
	info_fh.setFormatter(info_formatter)
	debug_fh.setFormatter(debug_formatter)

	logger.addHandler(ch)
	logger.addHandler(debug_fh)
	logger.addHandler(info_fh)

	return logger

def batch_to_device(batch, target_device):
	"""
	send a pytorch batch to a device (CPU/GPU)
	"""
	if isinstance(batch, dict):
		for key in batch:
			if torch.is_tensor(batch[key]):
				batch[key] = batch[key].to(target_device)
		return batch
	else:
		a = []
		for i in batch:
			a.append(i.to(target_device))

		return a[0], a[1], a[2]