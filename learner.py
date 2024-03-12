import os
import copy
import numpy as np
import torch
import torch.optim as optim
import torchaudio as audio
from abc import ABC, abstractmethod
from tqdm import tqdm, trange
from common.utils import detect_infnan
import common.utils_module as at_module
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
		
class Learner(ABC):
	""" Abstract class to the routines to train the one model and perform inferences
	"""
	def __init__(self, model):
		self.model = model
		self.max_score = -np.inf
		self.early_stop_counter = 0
		self.use_amp = False
		self.start_epoch = 1
		#self.device = device
		super().__init__()

	def mul_gpu(self):
		""" Use mutiple GPUs
		"""
		self.model = torch.nn.DataParallel(self.model) 
		# loss.mean() is only suitable for the case when batch size is divisible by the number of gpus, and when model output a value (e.g., loss).
		# When multiple gpus are used, 'module.' is added to the name of model parameters. 
		# So whether using one gpu or multiple gpus should be consistent for model pretraning and checkpoints loading.

	def cuda(self):
		""" Move the model to the GPU and perform the training and inference there
		"""
		self.model.cuda()
		# self.model = torch.compile(self.model)
		self.device = "cuda"

	def cpu(self):
		""" Move the model back to the CPU and perform the training and inference here
		"""
		self.model.cpu()
		self.device = "cpu"

	def amp(self):
		""" Use Automatic Mixed Precision to train network
		"""
		self.use_amp = True
		self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

	@abstractmethod
	def data_preprocess(self, mic_sig_batch=None, gt_batch=None):
		""" Preprocess microphone signals before trianing
		"""
		pass

	@abstractmethod
	def loss(self, pred_batch, gt_batch):
		""" Formulate downstream loss
		"""
		pass

	@abstractmethod
	def evaluate(self, pred_batch, gt_batch):
		""" Evaluate downstream task
		"""
		pass

	@abstractmethod
	def pretrain_evaluate(self, pred_batch, gt_batch):
		""" Evaluate pre-training pretext task
		"""
		pass

	def pretrain_epoch(self, dataset, lr=0.0001, epoch=None, return_diff=True):
		""" Train the model with an epoch of the dataset.
		"""
		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  
		optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0) # 5e-7
		
		# for param_group in optimizer.param_groups:
		# 	print(param_group['lr'])

		loss = 0
		if return_diff:
			diff = 0

		optimizer.zero_grad()
		pbar = tqdm(enumerate(dataset), total=len(dataset), leave=False) 

		for batch_idx, (mic_sig_batch, gt_batch) in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch))

			in_batch, = self.data_preprocess(mic_sig_batch, None)

			with torch.cuda.amp.autocast(enabled=self.use_amp):
				loss_batch, diff_batch, vis_batch = self.model(in_batch) # .contiguous()
			loss_batch = loss_batch.mean() # for multiple gpus
			diff_batch = diff_batch.mean()

			if self.use_amp:
				self.scaler.scale(loss_batch).backward()
				self.scaler.step(optimizer)
				self.scaler.update()
			else:
				loss_batch.backward()
				optimizer.step()

			optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss_batch.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (batch_idx + 1)))
			# pbar.set_postfix(loss=loss.item())
			pbar.update()

			loss += loss_batch.item()
			if return_diff: 
				diff += diff_batch.item()

		loss /= len(pbar)
		if return_diff: 
			diff /= len(pbar)

		if return_diff: 
			return loss, diff, vis_batch
		else:
			return loss

	def pretest_epoch(self, dataset, return_diff=True, return_eval=False):
		""" Test the model with an epoch of the dataset
		"""
		self.model.eval()  
		with torch.no_grad():
			loss = 0
			if return_diff: 
				diff = 0

			for mic_sig_batch, gt_batch in dataset:
				
				if gt_batch=={}:
					in_batch, = self.data_preprocess(mic_sig_batch, None)
				else:
					in_batch, dp_batch = self.data_preprocess(mic_sig_batch, torch.sum(gt_batch, -1))

				with torch.cuda.amp.autocast(enabled=self.use_amp):
					loss_batch, diff_batch, vis_batch = self.model(in_batch) 

				if gt_batch!={}:
					vis_batch['dp_tar'] = dp_batch.permute(0, 2, 3, 4, 1)  # (nb, nf, nt, 2, nmic)

				loss_batch = loss_batch.mean() 
				diff_batch = diff_batch.mean()

				loss += loss_batch.item()
				if return_diff: 
					diff += diff_batch.item()

			loss /= len(dataset)
			if return_diff: 
				diff /= len(dataset)

			if return_diff: 
				if return_eval:
					result_batch = self.pretrain_evaluate(pred_batch=vis_batch['pred'], gt_batch=vis_batch['tar'], mask_batch=vis_batch['mask'])
					return loss, diff, vis_batch, result_batch # only return visual data of the last batch
				else:
					return loss, diff, vis_batch 
			else:
				return loss


	def train_epoch(self, dataset, lr=0.0001, epoch=None, return_metric=False):
		""" Train the model with an epoch of the dataset
		"""

		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  
		optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
		# optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

		loss = 0
		if return_metric: 
			metric = 0

		optimizer.zero_grad()
		pbar = tqdm(enumerate(dataset), total=len(dataset), leave=False) 
		for batch_idx, (mic_sig_batch, gt_batch) in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch))

			in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

			with torch.cuda.amp.autocast(enabled=self.use_amp):
				pred_batch, embed_batch = self.model(in_batch)
				loss_batch = self.loss(pred_batch = pred_batch, gt_batch = gt_batch)

			if self.use_amp:
				self.scaler.scale(loss_batch).backward()
				self.scaler.step(optimizer)
				self.scaler.update()
			else:
				loss_batch.backward()
				optimizer.step()

			optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss_batch.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (batch_idx + 1)))
			# pbar.set_postfix(loss=loss.item())
			pbar.update()

			loss += loss_batch.item()

			if return_metric: 
				metric_batch = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch)
				metric += metric_batch 

		loss /= len(pbar)
		if return_metric: 
			metric /= len(pbar)

		if return_metric: 
			return loss, metric
		else:
			return loss

	def test_epoch(self, dataset, return_metric=False, return_vis=False):
		""" Test the model with an epoch of the dataset
		"""
		self.model.eval()  
		with torch.no_grad():
			loss = 0
			if return_metric: 
				metric = 0
			if return_vis:
				embed = []
				gt = []
			for mic_sig_batch, gt_batch in dataset:
				in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

				with torch.cuda.amp.autocast(enabled=self.use_amp):
					pred_batch, embed_batch = self.model(in_batch)
					loss_batch = self.loss(pred_batch=pred_batch, gt_batch=gt_batch)

				loss += loss_batch.item()

				if return_metric: 
					metric_batch = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch)
					metric += metric_batch  
				if return_vis:
					embed += [embed_batch]
					gt += [gt_batch]

			loss /= len(dataset)
			if return_metric: 
				metric /= len(dataset)

			if return_vis:
				vis_data = {'embed': torch.cat(embed, dim=0), 'label': torch.cat(gt, dim=0)}

			if return_metric: 
				if return_vis:
					return loss, metric, vis_data
				else:
					return loss, metric
			else:
				if return_vis:
					return loss, vis_data
				else:
					return loss 
				
	def test_epoch_T60(self, dataset, return_metric=False, return_vis=False):
		""" Test the model with an epoch of the dataset
		"""
		self.model.eval()  
		with torch.no_grad():
			loss = 0
			if return_metric: 
				metric = 0
			if return_vis:
				embed = []
				gt = []
			pbar = tqdm(enumerate(dataset), total=len(dataset), leave=False) 
			for batch_idx, (mic_sig_batch, gt_batch) in pbar:
				in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

				with torch.cuda.amp.autocast(enabled=self.use_amp):
					pred_batch, embed_batch = self.model(in_batch)
					# print(str(int(gt_batch.detach().cpu().numpy()[0][0][0]))+'st-result: ', pred_batch.detach().cpu().numpy()[0])
					# print('result: ', pred_batch.detach().cpu().numpy()[0],gt_batch.detach().cpu().numpy()[0],abs(pred_batch.detach().cpu().numpy()[0]-gt_batch.detach().cpu().numpy()[0]),snr.detach().cpu().numpy()[0])
					loss_batch = self.loss(pred_batch=pred_batch, gt_batch=gt_batch)

				loss += loss_batch.item()

				if return_metric: 
					metric_batch = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch)
					metric += metric_batch #.item()
				if return_vis:
					embed += [embed_batch]
					gt += [gt_batch]

			loss /= len(dataset)
			if return_metric: 
				metric /= len(dataset)

			if return_vis:
				vis_data = {'embed': torch.cat(embed, dim=0), 'label': torch.cat(gt, dim=0)}

			if return_metric: 
				if return_vis:
					return loss, metric, vis_data
				else:
					return loss, metric
			else:
				if return_vis:
					return loss, vis_data
				else:
					return loss 
				
	def test_epoch_DOA(self, dataset, return_metric=False, return_vis=False):
		""" Test the model with an epoch of the dataset
		"""
		self.model.eval()  
		with torch.no_grad():
			loss = 0
			if return_metric: 
				metric = 0
			if return_vis:
				embed = []
				gt = []
			for mic_sig_batch, gt_batch in dataset:
				in_batch, gt_batch = self.data_preprocess(mic_sig_batch, gt_batch)

				with torch.cuda.amp.autocast(enabled=self.use_amp):
					pred_batch, embed_batch = self.model(in_batch)
					loss_batch = self.loss(pred_batch=pred_batch, gt_batch=gt_batch)

				loss += loss_batch.item()

				if return_metric: 
					metric_batch = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch)
					metric += metric_batch 
				if return_vis:
					embed += [embed_batch]
					gt += [gt_batch]

			loss /= len(dataset)
			if return_metric: 
				metric /= len(dataset)

			if return_vis:
				vis_data = {'embed': torch.cat(embed, dim=0), 'label': torch.cat(gt, dim=0)}

			if return_metric: 
				if return_vis:
					return loss, metric, vis_data
				else:
					return loss, metric
			else:
				if return_vis:
					return loss, vis_data
				else:
					return loss 

	def smooth_data(self, data_list, alpha=0.8):
		""" current_smooth_data = alpha * previous_data + (1-alpha) * current_data
		"""
		nepoch = len(data_list)
		data_smooth_list = copy.deepcopy([data_list[0]])
		data_smooth = copy.deepcopy(data_list[0])
		for idx in range(1, nepoch):
			data_smooth = alpha * data_smooth + (1-alpha) * data_list[idx]
			data_smooth_list.append(data_smooth)

		return data_smooth_list

	def early_stopping(self, current_score, patience=5):
		""" stops when score is not declining for #patience epochs
		"""
		
		if current_score>= self.max_score:
			self.max_score = current_score
			self.early_stop_counter = 0
			stop_flag = False
			is_best_epoch = True
		else:
			self.early_stop_counter += 1
			if self.early_stop_counter >= patience:
				stop_flag = True
			else:
				stop_flag = False
			is_best_epoch = False
			
		return stop_flag, is_best_epoch

	def ensembling(self, checkpoints_dir, epochs):
		""" enseble the models of epochs
		"""
		nepoch_ave = len(epochs)
		state_dicts = {}
		for idx in range(nepoch_ave):
			epoch = epochs[idx]

			best_model_path = checkpoints_dir + '/model' + str(epoch) + '.tar'

			assert os.path.exists(best_model_path), f"{best_model_path} does not exist, can not load best model."

			checkpoint = torch.load(best_model_path, map_location=self.device)

			state_dict_one_epoch = checkpoint["model"]
			for key in state_dict_one_epoch.keys():
				if idx == 0:
					state_dicts[key] = state_dict_one_epoch[key] * 1/nepoch_ave
				else:
					state_dicts[key] += state_dict_one_epoch[key] * 1/nepoch_ave

		self.model.load_state_dict(state_dicts)

		print(f"\t Saving ensembling model checkpoint...")

		state_dict = {
				"epoch": epochs,
				"model": self.model.state_dict()
			}
		torch.save(state_dict, checkpoints_dir + "/ensemble_model.tar")

	def is_best_epoch(self, current_score):
		""" Check if the current model got the best metric score
        """
		if current_score >= self.max_score:
			self.max_score = current_score
			is_best_epoch = True
		else:
			is_best_epoch = False

		return is_best_epoch

	def save_checkpoint(self, epoch, checkpoints_dir, is_best_epoch = False, save_extra_hist=False):
		""" Save checkpoint to "checkpoints_dir" directory, which consists of:
            - the epoch number
            - the best metric score in history
            - the optimizer parameters
            - the model parameters
        """

		print(f"\t Saving {epoch} epoch model checkpoint...")
		if self.use_amp:
			state_dict = {
				"epoch": epoch,
				"max_score": self.max_score,
				# "optimizer": self.optimizer.state_dict(),
				"scaler": self.scaler.state_dict(), 
				"model": self.model.state_dict()
			}
		else:
			state_dict = {
				"epoch": epoch,
				"max_score": self.max_score,
				# "optimizer": self.optimizer.state_dict(),
				"model": self.model.state_dict()
			}
		torch.save(state_dict, checkpoints_dir + "/latest_model.tar")
		if save_extra_hist:
			torch.save(state_dict, checkpoints_dir + "/model"+str(epoch)+".tar")

		if is_best_epoch:
			print(f"\t Found a max score in the {epoch} epoch, saving...")
			torch.save(state_dict, checkpoints_dir + "/best_model.tar")


	def resume_checkpoint(self, checkpoints_dir, from_latest = True, as_all_state = True, param_frozen = False, ex_key=''):
		"""Resume from the latest/best checkpoint.
		"""

		if from_latest:

			latest_model_path = checkpoints_dir + "/latest_model.tar"

			assert os.path.exists(latest_model_path), f"{latest_model_path} does not exist, can not load latest checkpoint."

			checkpoint = torch.load(latest_model_path, map_location=self.device)
			self.start_epoch = checkpoint["epoch"] + 1
			self.max_score = checkpoint["max_score"]
			# self.optimizer.load_state_dict(checkpoint["optimizer"])
			if self.use_amp:
				self.scaler.load_state_dict(checkpoint["scaler"])
			if as_all_state:
				self.model.load_state_dict(checkpoint["model"])
			else:
				partial_state_dict = checkpoint["model"]
				all_state_dict = self.model.state_dict()
				for key in partial_state_dict:
					if ex_key+key in all_state_dict:
						all_state_dict[ex_key+key] = partial_state_dict[key]
				self.model.load_state_dict(all_state_dict)

			print(f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

		else: 
			best_model_path = checkpoints_dir + "/best_model.tar"

			assert os.path.exists(best_model_path), f"{best_model_path} does not exist, can not load best model."

			checkpoint = torch.load(best_model_path, map_location=self.device)
			epoch = checkpoint["epoch"]
			# self.max_score = checkpoint["max_score"]
			# self.optimizer.load_state_dict(checkpoint["optimizer"])
			# self.scaler.load_state_dict(checkpoint["scaler"])
			if as_all_state:
				self.model.load_state_dict(checkpoint["model"])
			else:
				partial_state_dict = checkpoint["model"]
				all_state_dict = self.model.state_dict()
				match_key_cnt = 0
				for key in partial_state_dict:
					if ex_key+key in all_state_dict:
						all_state_dict[ex_key+key] = partial_state_dict[key]
						match_key_cnt += 1
				
				print('# matched keys: ', match_key_cnt) 
				assert match_key_cnt>1, 'loaded model parameters and original parameters unmatched~'
				self.model.load_state_dict(all_state_dict)

			if param_frozen:
				for key, value in self.model.named_parameters():
					key.replace(ex_key, '')
					if key in partial_state_dict: # all frozen
						value.requires_grad = False

			print(f"Best model at {epoch} epoch loaded.")
			return epoch
	
	def resume_checkpoint_ensemble(self, checkpoints_dir):
		"""Resume from the checkpoint of ensembled model.
		"""

		model_path = checkpoints_dir + "/ensemble_model.tar"

		assert os.path.exists(model_path), f"{model_path} does not exist, can not load the checkpoint of ensembled model."

		checkpoint = torch.load(model_path, map_location=self.device)
		epoch = checkpoint["epoch"]
		self.model.load_state_dict(checkpoint["model"])
	
		print(f"Model of {epoch} epoch loaded.")

	def resume_checkpoint_epoch(self, checkpoints_dir, epoch):
		"""Resume from the checkpoint of specific epoch.
		"""

		model_path = checkpoints_dir + "/model"+str(epoch)+".tar"

		assert os.path.exists(model_path), f"{model_path} does not exist, can not load the checkpoint of specific epoch."

		checkpoint = torch.load(model_path, map_location=self.device)
		epoch0 = checkpoint["epoch"]
		assert epoch==epoch0, 'loaded epoch wrong~'
		self.model.load_state_dict(checkpoint["model"])
	
		print(f"Model of {epoch} epoch loaded.")

	def remove_checkpoint_epochs(self, checkpoints_dir, epochs):
		"""Remove the checkpoints of specific epochs.
		"""
		for epoch in epochs:
			model_path = checkpoints_dir + "/model"+str(epoch)+".tar"
			os.remove(model_path)

class STFTLearner(Learner):
	""" Learner for models which use STFTs of multi-channel microphone singals as input
	"""
	def __init__(self, model, win_len, win_shift_ratio, nfft, fre_used_ratio, fs, mel_scale=False, task=None, ch_mode='M'):
		super().__init__(model)

		self.ch_mode = ch_mode
		self.stft = at_module.STFT(
			win_len=win_len, 
			win_shift_ratio=win_shift_ratio, 
			nfft=nfft
			)
		self.istft = at_module.ISTFT(
			win_len=win_len, 
			win_shift_ratio=win_shift_ratio, 
			nfft=nfft,
			inv=False
			)
		self.mel_scale = mel_scale
		if mel_scale:
			self.mel_transform = audio.transforms.MelScale(
				n_mels=30,
				sample_rate=fs,
				f_min=0,
				f_max=fs//2,
				n_stft=nfft//2+1,
				)
		if fre_used_ratio == 1:
			self.fre_range_used = range(1, int(nfft/2*fre_used_ratio)+1, 1)
		elif fre_used_ratio == 0.5:
			self.fre_range_used = range(0, int(nfft/2*fre_used_ratio), 1)
		else:
			raise Exception('Prameter fre_used_ratio unexpected')
		# self.addbatch = at_module.AddChToBatch(ch_mode=self.ch_mode)
		# self.removebatch = at_module.RemoveChFromBatch(ch_mode=self.ch_mode)
		self.task = task

	def data_preprocess(self, mic_sig_batch=None, gt_batch=None, eps=1e-6):
		""" Preprocess microphone signals before model training
			Args: 		mic_sig_batch - (nbatch, nsample, nch)
						gt_batch - {'TDOA':, 'DRR':, ...}}
			Returns: 	[reim_rebatch - (nb,nch,nf,nt,2), gt_batch - (nbatch, ...)]
		"""
		data = []
		if mic_sig_batch is not None:
			mic_sig_batch = mic_sig_batch.to(self.device)
			stft = self.stft(signal = mic_sig_batch) 	# (nb,nf,nt,nch)
			stft = stft.permute(0, 3, 1, 2)  # (nb,nch,nf,nt)

			nor_flag = True
			if nor_flag:
				mag = torch.abs(stft[:,0:1,:,:])
				mean_value = torch.mean(mag.reshape(mag.shape[0],-1), dim=1)
				mean_value = mean_value[:,np.newaxis,np.newaxis,np.newaxis].expand(mag.shape)
				stft = stft/(mean_value+eps) # (nb*(nch-1),nch_pair,nf,nt) = (nb,nch,nf,nt) when nch=2

			# Change batch for multi-channel data (nb,nch,nf,nt)→(nb*(nch-1),2,nf,nt)/(nb*(nch-1)*nch/2,2,nf,nt)
			# stft_rebatch = self.addbatch(stft)
			# stft_rebatch = stft  # (nb*(nch-1),nch_pair,nf,nt)
			# reim_rebatch = torch.view_as_real(stft_rebatch) # (nb*(nch-1),nch_pair,nf,nt,2)

			reim_rebatch = torch.view_as_real(stft) # (nb,nch,nf,nt,2)

			if self.mel_scale:
				reim_rebatch = self.mel_transform(reim_rebatch.permute(0,1,4,2,3).to('cpu')).permute(0,1,3,4,2).contiguous().to(stft_rebatch.device) # (nb,nch,nmel,nt,2)
			else:
				reim_rebatch = reim_rebatch[:, :, self.fre_range_used, :, :]

			data += [reim_rebatch]

		if gt_batch is not None:
			if (mic_sig_batch is not None) & (isinstance(gt_batch, type(mic_sig_batch))): # gt_batch = dp_mic_signal
				if nor_flag:
					dp_mic_sig_batch = gt_batch.to(self.device)
					dp_stft = self.stft(signal = dp_mic_sig_batch) 	# (nb,nf,nt,nch)
					dp_stft = dp_stft.permute(0, 3, 1, 2)  # (nb,nch,nf,nt)
					dp_stft = dp_stft/(mean_value+eps)
					dp_stft_rebatch = dp_stft  # (nb*(nch-1),nch_pair,nf,nt)
					dp_reim_rebatch = torch.view_as_real(dp_stft_rebatch) # (nb*(nch-1),nch_pair,nf,nt,2
					data += [dp_reim_rebatch.to(self.device)]
				else:
					data += [gt_batch.to(self.device)]
			else:
				gt_batch = gt_batch[self.task].to(self.device)
				gt_batch = self.get_tar_batch(gt_batch=gt_batch)
				data += [ gt_batch ]

		return data # [Input, TDOA/T60/DRR/DOA/C50/C80]
	
	def pretrain_evaluate(self, pred_batch, gt_batch, mask_batch):
		""" Evaluate the performance of pre-training (namely the signal reconstrution performance of the pretext task)
			Args:	pred_batch: (nb,nf,nt,nreim,nch)
					gt_batch: (nb,nf,nt,nreim,nch)
					mask_batch: (nb,nf,nt,nch)
			Returns: {'sig_pred', 'sig_tar', 'mse', 'pesq', 'pesq_mask_ch'}
		"""
		pred = pred_batch.permute(0, 1, 2, 4, 3).contiguous() # (nb,nf,nt,nch,nreim)
		stft_pred = torch.view_as_complex(pred) # (nb,nf,nt,nch)
		stft_pred = torch.cat((torch.zeros_like(stft_pred[:, 0:1, :, :]), stft_pred), dim=1)
		sig_pred = self.istft(stft_pred) # (nb,nsample,nch)
		sig_pred = sig_pred/torch.max(sig_pred)

		gt = gt_batch.permute(0, 1, 2, 4, 3).contiguous() # (nb,nf,nt,nch,nreim)
		stft_gt = torch.view_as_complex(gt) # (nb,nf,nt,nch)
		stft_gt = torch.cat((torch.zeros_like(stft_gt[:,0:1,:,:]), stft_gt), dim=1)
		sig_gt = self.istft(stft_gt) # (nb,nsample,nch)
		sig_gt = sig_gt/torch.max(sig_gt)

		mask_dense = mask_batch[:, :, :, np.newaxis, :].tile(1, 1, 1, 2, 1)
		diff = (pred_batch-gt_batch)**2

		diff_mask = diff * (1-mask_dense)
		mse_mask = torch.sum(diff_mask) / torch.sum(1-mask_dense) # loss

		diff_mask_ch = torch.sum(diff_mask, dim=4)
		mse_mask_ch = torch.mean(diff_mask_ch)

		mse = torch.mean(diff)

		fs = 16000
		nb, _, _, nch = mask_batch.shape
		pesq_mask_ch = torch.zeros(nb)
		pesq = torch.zeros(nb, nch)
		for b_idx in range(nb):
			if torch.sum(torch.sum(mask_batch[b_idx, :, :, 1], dim=1), dim=0)>torch.sum(torch.sum(mask_batch[b_idx, :, :, 0], dim=1), dim=0):
				mask_ch_idx = 0
			else:
				mask_ch_idx = 1
			for ch_idx in range(nch):
				pesq[b_idx, ch_idx] = perceptual_evaluation_speech_quality(sig_pred[b_idx, :, ch_idx], sig_gt[b_idx, :, ch_idx], fs, 'wb')
			pesq_mask_ch = pesq[:, mask_ch_idx]
		result = {'sig_pred':sig_pred, 'sig_tar':sig_gt, 'mse':mse, 'mse_mask':mse_mask, 'mse_mask_ch':mse_mask_ch, 'pesq':pesq, 'pesq_mask_ch':pesq_mask_ch}

		return result
	
	def get_tar_batch(self, gt_batch):
		""" Calculate the loss for downstream training
			Args: gt_batch - (nbatch, ...)
			Returns: tar_batch - (nbatch, 1)
		"""
		if self.task == 'TDOA':
			TDOAw_batch = gt_batch*16000  # (nb*nseg*nch-1*nsources)
			tar_batch = torch.mean(TDOAw_batch[:, :, 0, 0:1], dim=1) 
		elif self.task == 'T60':
			T60_batch = gt_batch*1  # (nb)
			tar_batch = T60_batch[:,np.newaxis]
		elif self.task == 'DRR':
			DRRw_batch = gt_batch*1   # (nb*nseg*nsources)
			tar_batch = torch.mean(DRRw_batch[: ,: , 0:1], dim=1) 
		elif self.task == 'DOA':
			DOAw_batch = gt_batch  # (nb*nseg*2*nsources)
			tar_batch = torch.mean(DOAw_batch[:, :, 0:1, 0], dim=1) # (nb*1) for azimuth estimation
		elif self.task == 'C50':
			C50w_batch = gt_batch  # (nb*nseg*nsources)
			tar_batch = torch.mean(C50w_batch[:, :, 0:1], dim=1) 
		elif self.task == 'C80':
			C80w_batch = gt_batch  # (nb*nseg*nsources)
			tar_batch = torch.mean(C80w_batch[:, :, 0:1], dim=1) 
		elif self.task == 'SNR':
			SNR_batch = gt_batch*1 # (nb)
			tar_batch = SNR_batch[:, np.newaxis]
		elif self.task == 'ABS':
			abs_batch = gt_batch*1 # (nb)
			tar_batch = abs_batch[:, np.newaxis]
		elif self.task == 'SUR':
			S_batch = torch.log10(gt_batch*1) # (nb)
			tar_batch = S_batch[:, np.newaxis]
		elif self.task == 'VOL':
			V_batch = torch.log10(gt_batch*1) # (nb)
			tar_batch = V_batch[:, np.newaxis]
		else:
			raise Exception('Task mode unrecognized')
		return tar_batch

	def loss(self, pred_batch, gt_batch):
		""" Calculate the loss for downstream training
			Args: 	pred_batch - (nbatch, 1 )
					gt_batch - (nbatch, 1)
			Returns: loss
        """ 
		if self.task != 'DOA':
			loss = torch.nn.functional.mse_loss(pred_batch.contiguous(), gt_batch.contiguous().detach())
			# loss = torch.mean(torch.abs(pred_batch.contiguous()-tar_batch.contiguous().detach()))
		# else:
		# 	doa_label = torch.round(gt_batch).long()
		# 	loss = torch.nn.functional.cross_entropy(pred_batch, doa_label)

		return loss

	def evaluate(self, pred_batch, gt_batch):
		""" Evaluate the performance (MAE) of downstream tasks
			Args: 	pred_batch - (nbatch, 1 )
					gt_batch - (nbatch, 1)
			Returns: MAE
        """
		if self.task != 'DOA':
			mae = torch.mean(torch.abs(pred_batch.contiguous().detach()-gt_batch.contiguous().detach()))

		return mae
	
	def mae_wotrain(self, dataset_train, dataset_test):
		""" Calculate the MAE for the traning and test data, when treating the training mean as prediction
		"""
		idx = 0 
		for _, gt_batch in dataset_train:
			gt_ = gt_batch[self.task].to(self.device)
			tar_batch = self.get_tar_batch(gt_batch=gt_)
			if idx == 0:
				gt = tar_batch
			else:
				gt = torch.concat((gt, tar_batch), dim=0)
			idx = idx+1
		min = torch.min(gt, dim=0)[0].item()
		max = torch.max(gt, dim=0)[0].item()
		mean = torch.mean(gt).item()
		mae = torch.mean(torch.abs(gt-mean)).item()

		idx = 0 
		for _, gt_batch in dataset_test:
			gt_ = gt_batch[self.task].to(self.device)
			tar_batch = self.get_tar_batch(gt_batch=gt_)
			if idx == 0:
				gt_test = tar_batch
			else:
				gt_test = torch.concat((gt_test, tar_batch), dim=0)
			idx = idx+1

		mae_test = torch.mean(torch.abs(gt_test-mean)).item()
		min_test = torch.min(gt_test, dim=0)[0].item()
		max_test = torch.max(gt_test, dim=0)[0].item()

		return mae_test, min_test, max_test, mae, mean, min, max

class JointSTFTLearner(Learner):
	""" Joint learner for models which use STFTs of multi-channel microphone singals as input (multi-task learning)
	"""
	def __init__(self, model, win_len, win_shift_ratio, nfft, fre_used_ratio, fs, mel_scale=False, task=None, ch_mode='M'):
		super().__init__(model)

		self.ch_mode = ch_mode
		self.stft = at_module.STFT(
			win_len=win_len, 
			win_shift_ratio=win_shift_ratio, 
			nfft=nfft
			)
		self.mel_scale = mel_scale
		if mel_scale:
			self.mel_transform = audio.transforms.MelScale(
				n_mels=30,
				sample_rate=fs,
				f_min=0,
				f_max=fs//2,
				n_stft=nfft//2+1,
				)
		if fre_used_ratio == 1:
			self.fre_range_used = range(1, int(nfft/2*fre_used_ratio)+1, 1)
		elif fre_used_ratio == 0.5:
			self.fre_range_used = range(0, int(nfft/2*fre_used_ratio), 1)
		else:
			raise Exception('Prameter fre_used_ratio unexpected')
		# self.addbatch = at_module.AddChToBatch(ch_mode=self.ch_mode)
		# self.removebatch = at_module.RemoveChFromBatch(ch_mode=self.ch_mode)
		self.task = task

	def data_preprocess(self, mic_sig_batch=None, gt_batch=None, eps=1e-6):
		data = []
		if mic_sig_batch is not None:
			mic_sig_batch = mic_sig_batch.to(self.device)
			stft = self.stft(signal = mic_sig_batch) 	# (nb,nf,nt,nch)
			stft = stft.permute(0, 3, 1, 2)  # (nb,nch,nf,nt)

			nor_flag = True
			if nor_flag:
				mag = torch.abs(stft[:,0:1,:,:])
				mean_value = torch.mean(mag.reshape(mag.shape[0],-1), dim=1)
				mean_value = mean_value[:,np.newaxis,np.newaxis,np.newaxis].expand(mag.shape)
				stft = stft/(mean_value+eps) # (nb*(nch-1),nch_pair,nf,nt) = (nb,nch,nf,nt) when nch=2

			## Change batch for multi-channel data (nb,nch,nf,nt)→(nb*(nch-1),2,nf,nt)/(nb*(nch-1)*nch/2,2,nf,nt)
			# stft_rebatch = self.addbatch(stft)
			# stft_rebatch = stft  # (nb*(nch-1),nch_pair,nf,nt)
			# reim_rebatch = torch.view_as_real(stft_rebatch) # (nb*(nch-1),nch_pair,nf,nt,2)

			reim_rebatch = torch.view_as_real(stft) # (nb,nch,nf,nt,2)
			
			if self.mel_scale:
				reim_rebatch = self.mel_transform(reim_rebatch.permute(0,1,4,2,3).to('cpu')).permute(0,1,3,4,2).contiguous().to(stft_rebatch.device) # (nb,nch,nmel,nt,2)
			else:
				reim_rebatch = reim_rebatch[:, :, self.fre_range_used, :, :]

			data += [reim_rebatch]
		
		if gt_batch is not None:

			gt = {}
			task = self.task.split('-')
			keys = ['TDOA', 'DRR', 'T60', 'C50', 'SNR', 'ABS', 'SUR', 'VOL']
			for key in keys:
				if (key in task) & (key in gt_batch.keys()):
					gt[key] = gt_batch[key].to(self.device)

			data += [ gt ]

			assert (len(task)==len(gt.keys())) | (len(gt.keys())==1), 'task or gt error'

		return data # [Input, TDOA/T60/DRR/DOA/C50/C80]

	def loss(self, pred_batch, gt_batch):
		loss = 0
		task = self.task.split('-')

		pred_task_idx = -1
		gt_task_idx = -1
		if 'TDOA' in task:
			pred_task_idx += 1
			if 'TDOA' in gt_batch.keys():
				gt_task_idx += 1
				TDOAw_batch = gt_batch['TDOA']*16000  # (nb*nseg*nch-1*nsources)
				tar_TDOA = torch.mean(TDOAw_batch[:,:,0,0:1], dim=1) 
				loss += torch.nn.functional.mse_loss(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous(), tar_TDOA.contiguous().detach())
		if 'DRR' in task:
			pred_task_idx += 1
			if 'DRR' in gt_batch.keys():
				gt_task_idx += 1
				DRRw_batch = gt_batch['DRR']*1   # (nb*nseg*nsources)
				tar_DRR = torch.mean(DRRw_batch[:,:,0:1], dim=1) 
				loss += torch.nn.functional.mse_loss(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous(), tar_DRR.contiguous().detach()) 
		if 'T60' in task:
			pred_task_idx += 1
			if 'T60' in gt_batch.keys():
				gt_task_idx += 1
				T60_batch = gt_batch['T60']*1  # (nb)
				tar_T60 = T60_batch[:,np.newaxis]
				loss += torch.nn.functional.mse_loss(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous(), tar_T60.contiguous().detach())
		if 'C50' in task:
			pred_task_idx += 1
			if 'C50' in gt_batch.keys():
				gt_task_idx += 1
				C50w_batch = gt_batch['C50']*1  # (nb*nseg*nsources)
				tar_C50 = torch.mean(C50w_batch[:,:,0:1], dim=1) 
				loss += torch.nn.functional.mse_loss(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous(), tar_C50.contiguous().detach())
		if 'ABS' in task:
			pred_task_idx += 1
			if 'ABS' in gt_batch.keys():
				gt_task_idx += 1
				abs_batch = gt_batch['ABS']*1 # (nb)
				tar_ABS = abs_batch[:,np.newaxis]
				loss += torch.nn.functional.mse_loss(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous(), tar_ABS.contiguous().detach())
		if 'SNR' in task:
			pred_task_idx += 1
			if 'SNR' in gt_batch.keys():
				gt_task_idx += 1
				SNR_batch = gt_batch['SNR']*1 # (nb)
				tar_SNR = SNR_batch[:,np.newaxis]
				loss += torch.nn.functional.mse_loss(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous(), tar_SNR.contiguous().detach())
		if 'SUR' in task:
			pred_task_idx += 1
			if 'SUR' in gt_batch.keys():
				gt_task_idx += 1
				S_batch = torch.log10(gt_batch['SUR']*1) # (nb)
				tar_SUR = S_batch[:,np.newaxis]
				loss += torch.nn.functional.mse_loss(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous(), tar_SUR.contiguous().detach())
		if 'VOL' in task:
			pred_task_idx += 1
			if 'VOL' in gt_batch.keys():
				gt_task_idx += 1
				V_batch = torch.log10(gt_batch['VOL']*1) # (nb)
				tar_VOL = V_batch[:,np.newaxis]
				loss += torch.nn.functional.mse_loss(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous(), tar_VOL.contiguous().detach())
		loss = loss/(gt_task_idx+1)

		return loss

	def evaluate(self, pred_batch, gt_batch):
		task = self.task.split('-')
		pred_task_idx = -1
		gt_task_idx = -1
		mae = torch.zeros((len(task))).to(pred_batch.device)
		if 'TDOA' in task:
			pred_task_idx += 1
			if 'TDOA' in gt_batch.keys():
				gt_task_idx += 1
				TDOAw_batch = gt_batch['TDOA']*16000  # (nb*nseg*nch-1*nsources)
				tar_TDOA = torch.mean(TDOAw_batch[:,:,0,0:1], dim=1) 
				mae[pred_task_idx] = torch.mean(torch.abs(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous().detach()-tar_TDOA.contiguous().detach()))
		if 'DRR' in task:
			pred_task_idx += 1
			if 'DRR' in gt_batch.keys():
				gt_task_idx += 1
				DRRw_batch = gt_batch['DRR']*1   # (nb*nseg*nsources)
				tar_DRR = torch.mean(DRRw_batch[:,:,0:1], dim=1)
				mae[pred_task_idx] = torch.mean(torch.abs(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous().detach()-tar_DRR.contiguous().detach()))
		if 'T60' in task: 
			pred_task_idx += 1
			if 'T60' in gt_batch.keys():
				gt_task_idx += 1
				T60_batch = gt_batch['T60']*1  # (nb)
				tar_T60 = T60_batch[:,np.newaxis]
				mae[pred_task_idx] = torch.mean(torch.abs(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous().detach()-tar_T60.contiguous().detach()))
		if 'C50' in task:
			pred_task_idx += 1
			if 'C50' in gt_batch.keys():
				gt_task_idx += 1
				C50w_batch = gt_batch['C50']*1  # (nb*nseg*nsources)
				tar_C50 = torch.mean(C50w_batch[:,:,0:1], dim=1)
				mae[pred_task_idx] = torch.mean(torch.abs(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous().detach()-tar_C50.contiguous().detach()))
		if 'ABS' in task:
			pred_task_idx += 1
			if 'ABS' in gt_batch.keys():
				gt_task_idx += 1
				abs_batch = gt_batch['ABS']*1 # (nb)
				tar_ABS = abs_batch[:,np.newaxis]
				mae[pred_task_idx] = torch.mean(torch.abs(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous().detach()-tar_ABS.contiguous().detach()))
		if 'SNR' in task: 
			pred_task_idx += 1
			if 'SNR' in gt_batch.keys():
				gt_task_idx += 1
				SNR_batch = gt_batch['SNR']*1 # (nb)
				tar_SNR = SNR_batch[:,np.newaxis]
				mae[pred_task_idx] = torch.mean(torch.abs(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous().detach()-tar_SNR.contiguous().detach()))
		if 'SUR' in task:
			pred_task_idx += 1
			if 'SUR' in gt_batch.keys():
				gt_task_idx += 1
				S_batch = torch.log10(gt_batch['SUR']*1) # (nb)
				tar_SUR = S_batch[:,np.newaxis]
				mae[pred_task_idx] = torch.mean(torch.abs(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous().detach()-tar_SUR.contiguous().detach()))
		if 'VOL' in task:
			pred_task_idx += 1
			if 'VOL' in gt_batch.keys():
				gt_task_idx += 1
				V_batch = torch.log10(gt_batch['VOL']*1) # (nb)
				tar_VOL = V_batch[:,np.newaxis]
				mae[pred_task_idx] = torch.mean(torch.abs(pred_batch[:, pred_task_idx:pred_task_idx+1].contiguous().detach()-tar_VOL.contiguous().detach()))
 
		return mae
