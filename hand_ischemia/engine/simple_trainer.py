import sys
import os
import torch
from hand_ischemia.config import convert_to_dict
import torcheval.metrics
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix, BinaryAUPRC
import torchmetrics.classification as classification
import mlflow
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
__all__ = ['SimpleTrainer']


class SimpleTrainer(object):

    def __init__(self, cfg):

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.epochs = cfg.DENOISER.EPOCHS
        self.BinaryAccuracy = classification.BinaryAccuracy().to(self.device)
        self.BinaryAUROC = classification.BinaryAUROC().to(self.device)
        self.BinaryAvgPrecision = classification.BinaryAveragePrecision().to(self.device)
        self.BinaryRecall = classification.BinaryRecall().to(self.device)
        self.BinaryF1Score = classification.BinaryF1Score().to(self.device)
        self.BinaryConfusionMatrix = classification.BinaryConfusionMatrix().to(self.device)
        self.BinaryPrecisionRecallCurve = classification.BinaryPrecisionRecallCurve().to(self.device)
        self.BinaryROC = classification.BinaryROC().to(self.device)
        
        self.BinaryAccuracy_eval = torcheval.metrics.BinaryAccuracy().to(self.device)
        #self.BinaryAUROC_eval = torcheval.metrics.BinaryAUROC().to(self.device)
        self.BinaryPrecision_eval = torcheval.metrics.BinaryPrecision().to(self.device)
        self.BinaryRecall_eval = torcheval.metrics.BinaryRecall().to(self.device)
        self.BinaryF1Score_eval = torcheval.metrics.BinaryF1Score().to(self.device)
        self.BinaryAUPRC_eval = torcheval.metrics.BinaryAUPRC().to(self.device)
        #self.BinaryConfusionMatrix = torcheval.metrics.BinaryConfusionMatrix().to(self.device)
        #self.BinaryPrecisionRecallCurve = torcheval.metrics.BinaryPrecisionRecallCurve().to(self.device)
        #self.BinaryROC = classification.BinaryROC().to(self.device)

    def log_config_dict(self, cfg):
        # Log the parameters
        cfg_dict = convert_to_dict(cfg)
        for key in cfg_dict.keys():
            mlflow.log_params(cfg_dict[key])

    def compute_metrics(self, pred_labels, gt_labels):
                
        acc = BinaryAccuracy().update(pred_labels, gt_labels).compute().item()
        auroc = BinaryAUROC().update(pred_labels, gt_labels).compute().item()
        precision = BinaryPrecision().update(pred_labels, gt_labels).compute().item()
        recall = BinaryRecall().update(pred_labels, gt_labels).compute().item()
        f1score = BinaryF1Score().update(pred_labels, gt_labels).compute().item()
        #confusion = BinaryConfusionMatrix().update(pred_labels, gt_labels).compute()
        
        metrics_dict = {'acc': acc, 'auroc': auroc, 'precision': precision, 'recall': recall, 'f1score': f1score, 'confusion':0}
        return metrics_dict
    '''   
    def compute_torchmetrics(self, pred_labels, gt_labels, epoch, mode='test'):
                
        acc = self.BinaryAccuracy(pred_labels, gt_labels).item()
        auroc = self.BinaryAUROC(pred_labels, gt_labels).item()
        precision = self.BinaryAvgPrecision(pred_labels, gt_labels.long()).item()
        recall = self.BinaryRecall(pred_labels, gt_labels).item()
        f1score = self.BinaryF1Score(pred_labels, gt_labels).item()
        #confusion = BinaryConfusionMatrix.to(self.device)(pred_labels, gt_labels).compute()
        
        if mode == 'test':
            metrics_dict = {'test_acc': acc, 'test_auroc': auroc, 'test_precision': precision,
                            'test_recall': recall, 'test_f1score': f1score, 'test_confusion':0}
        else:
            metrics_dict = {'train_acc': acc, 'train_auroc': auroc, 'train_precision': precision,
                            'train_recall': recall, 'train_f1score': f1score, 'train_confusion':0}
        if epoch == self.epochs:
            self.BinaryConfusionMatrix.update(pred_labels, gt_labels)
            fig, ax = self.BinaryConfusionMatrix.plot()
            mlflow.log_figure(fig, 'confusion_mat.jpg')
            
            self.BinaryPrecisionRecallCurve.update(pred_labels, gt_labels.long())
            fig, ax = self.BinaryPrecisionRecallCurve.plot()
            mlflow.log_figure(fig, 'pr_curve.jpg')
            
            self.BinaryROC.update(pred_labels, gt_labels.long())
            fig, ax = self.BinaryROC.plot()
            mlflow.log_figure(fig, 'roc_curve.jpg')
            
            plt.close()

        return metrics_dict
    '''
    def update_torchmetrics(self, pred_labels, gt_labels, pred_class, gt_class, mode='test'):
        
        #Update the metrics    
        self.BinaryAccuracy.update(pred_labels, gt_labels)
        self.BinaryAUROC.update(pred_labels, gt_labels)
        self.BinaryAvgPrecision.update(pred_labels, gt_labels.long())
        self.BinaryRecall.update(pred_labels, gt_labels)
        self.BinaryF1Score.update(pred_labels, gt_labels)   
        self.BinaryConfusionMatrix.update(pred_class, gt_class)
        self.BinaryPrecisionRecallCurve.update(pred_labels, gt_labels.long())
        self.BinaryROC.update(pred_labels, gt_labels.long())
        
        
        if pred_class.dim() == 0:
            pred_class = pred_class.unsqueeze(0)
            gt_class = gt_class.unsqueeze(0)
        #update torcheval metrics
        self.BinaryAccuracy_eval.update(pred_class, gt_class)
        #self.BinaryAUROC_eval.update(pred_labels, gt_labels)
        self.BinaryPrecision_eval.update(pred_class, gt_class)
        self.BinaryRecall_eval.update(pred_class, gt_class)
        self.BinaryF1Score_eval.update(pred_class, gt_class)   
        self.BinaryAUPRC_eval.update(pred_class, gt_class)
        #self.BinaryConfusionMatrix_eval.update(pred_class, gt_class)
        #self.BinaryPrecisionRecallCurve_eval.update(pred_labels, gt_labels.long())
        #self.BinaryROC_eval.update(pred_labels, gt_labels.long())
    
    def compute_torchmetrics(self, epoch, mode='test'):
        
        
        
        #Compute the metrics
        acc = self.BinaryAccuracy.compute().item()
        auroc = self.BinaryAUROC.compute().item()
        precision = self.BinaryAvgPrecision.compute().item()
        recall = self.BinaryRecall.compute().item()
        f1score = self.BinaryF1Score.compute().item()
        
        #Compute the metrics
        acc_tm = self.BinaryAccuracy_eval.compute().item()
        #auroc = self.BinaryAUROC_eval.compute().item()
        precision_tm = self.BinaryPrecision_eval.compute().item()
        recall_tm = self.BinaryRecall_eval.compute().item()
        f1score_tm = self.BinaryF1Score_eval.compute().item()
        auprc_tm = self.BinaryAUPRC_eval.compute().item()
        metrics_dict = {'test_acc': acc, 'test_auroc': auroc, 'test_precision': precision,
                        'test_recall': recall, 'test_f1score': f1score, 'test_confusion':0,
                        'acc_tm': acc_tm, 'recall_tm': recall_tm, 'precision_tm': precision_tm, 'f1_tm': f1score_tm,
                        'auprc': auprc_tm}
        
        
        if epoch == self.epochs:
            confusion_fig, ax = self.BinaryConfusionMatrix.plot()
            mlflow.log_figure(confusion_fig, 'confusion_mat.jpg')
            mlflow.log_figure(confusion_fig, 'confusion_mat.svg')
            precision_fig, ax = self.BinaryPrecisionRecallCurve.plot()
            mlflow.log_figure(precision_fig, 'pr_curve.jpg')
            mlflow.log_figure(precision_fig, 'pr_curve.svg')
            roc_fig, ax = self.BinaryROC.plot()
            mlflow.log_figure(roc_fig, 'roc_curve.jpg')
            mlflow.log_figure(roc_fig, 'roc_curve.svg')
            
            plt.close()
        
        #Reset the metrics
        
        self.BinaryAccuracy.reset()
        self.BinaryAUROC.reset()
        self.BinaryAvgPrecision.reset()
        self.BinaryRecall.reset()
        self.BinaryF1Score.reset()   
        self.BinaryConfusionMatrix.reset()
        self.BinaryPrecisionRecallCurve.reset()
        self.BinaryROC.reset()
        self.BinaryAccuracy_eval.reset()
        self.BinaryPrecision_eval.reset()
        self.BinaryRecall_eval.reset()
        self.BinaryF1Score_eval.reset()
        self.BinaryAUPRC_eval.reset()

        return metrics_dict
        
    def _compute_rmse_and_pte6(self, HR_gt, HR_est):
        HR_gt = np.array(HR_gt)
        HR_est = np.array(HR_est)
        N = len(HR_est)

        rmse = np.sqrt(np.mean(np.square(HR_gt - HR_est)))
        mae = np.mean(np.abs(HR_gt - HR_est))
        pte6 = 100 * np.sum(np.where(np.abs(HR_gt - HR_est) < 6, 1.0, 0.0)) / N

       
        metrics = {'rmse': rmse, 'mae': mae, 'pte6': pte6}
        
        return metrics
    
    def _calculate_SNR(self, pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
        
        """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
        """
        def _next_power_of_2(x):
            """Calculate the nearest power of 2."""
            return 1 if x == 0 else 2 ** (x - 1).bit_length()
        
        def mag2db(mag):
            """Convert magnitude to db."""
            return 20. * np.log10(mag)
        batch, reg, siglen = pred_ppg_signal.shape
        SNRs = []
        for i in range(0, reg):

            sig = pred_ppg_signal[0, i, :]
            # Get the first and second harmonics of the ground truth HR in Hz
            first_harmonic_freq = hr_label / 60
            second_harmonic_freq = 2 * first_harmonic_freq
            deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

            # Calculate FFT
            sig = np.expand_dims(sig, 0)
            N = _next_power_of_2(sig.shape[1])
            f_ppg, pxx_ppg = scipy.signal.periodogram(sig, fs=fs, nfft=N, detrend=False)

            # Calculate the indices corresponding to the frequency ranges
            idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
            idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
            idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
            & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
            & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

            # Select the corresponding values from the periodogram
            pxx_ppg = np.squeeze(pxx_ppg)
            pxx_harmonic1 = pxx_ppg[idx_harmonic1]
            pxx_harmonic2 = pxx_ppg[idx_harmonic2]
            pxx_remainder = pxx_ppg[idx_remainder]

            # Calculate the signal power
            signal_power_hm1 = np.sum(pxx_harmonic1)
            signal_power_hm2 = np.sum(pxx_harmonic2)
            signal_power_rem = np.sum(pxx_remainder)

            # Calculate the SNR as the ratio of the areas
            if not signal_power_rem == 0: # catches divide by 0 runtime warning 
                SNR = mag2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
            else:
                SNR = 0
            SNRs.append(SNR)
        return max(SNRs)
