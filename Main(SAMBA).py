import os
import sys

# Add the fm directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
from types import SimpleNamespace
from Models import EEGMS
import torch
from pytorch_lightning import seed_everything

from Experiments.Downstream_Fine import run_downstream_finetune
from Experiments.Downstream_Probing import run_linear_probe
from Experiments.Reconstruction import Reconstruction

from utility.data_loader import data_loader, fused_emotiv_loader


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 'yes', 'y')
# ------------------------
# Argument Parser
# ------------------------
parser = argparse.ArgumentParser()
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--use_compile', type=str2bool, default=False, help="Enable torch.compile() to accelerate training")
parser.add_argument('--num_workers', type=int, default=64, help='Number of DataLoader worker processes')
parser.add_argument('--result_root', type=str, default='./fm/EEGMS/Checkpoints', help='Root directory for saving results')
parser.add_argument('--save_log', type=str2bool, default=True, help='Whether to create logger or not')
# ----------------------------------------- Train Parameters ----------------------------------------------
parser.add_argument('--Training_mode', type=str, default='Reconstruction', choices=['Reconstruction', 'Linear_Probe', 'Fine_Tuning', 'Supervised', 'Nonlinear_Probe'], help='Training mode')
parser.add_argument('--model_name', type=str, default='EEGMS', choices=['EEGMS', 'EEGM2', 'UNetSAIE'], help='Model name')
parser.add_argument('--loss', type=str, default='L1Spectral', choices=['L1', 'Spectral', 'L1Spectral', 'L2Spectral', 'L1SpectralERP'])
parser.add_argument("--masking_method", type=str, default="TSR", choices=["No_masking", "Random", "SSP", "TSR"], help="Masking method for training")
parser.add_argument('--time_mask_ratio', type=float, default=0.5, help='Masking ratio for time domain') 
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--early_stop', type=int, default=350)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--lr', type=float, default=5e-5)
# OneCyle parameters
parser.add_argument('--weight_decay', type=float, default=1e-2, help='OneCyle weight decay')
parser.add_argument('--max_lr', type=float, default=5e-4, help='OneCyle max learning rate')
parser.add_argument('--div_factor', type=float, default=2, help='OneCyle div factor for learning rate')
parser.add_argument('--anneal_strategy', type=str, default='cos', help='OneCyle anneal strategy for learning rate')
parser.add_argument('--pct_start', type=float, default=0.1, help='OneCyle percentage of total steps for increasing learning rate')
parser.add_argument('--final_div_factor', type=float, default=100, help='OneCyle final div factor for learning rate')
# ----------------------------------------- Checkpoing Setting --------------------------------------------------------
parser.add_argument('--monitor_metric', type=str, default='Val/Loss', choices=['val_loss', 'train_loss', 'val_accuracy', 'val_mse'], help='Metric to monitor for checkpointing')
parser.add_argument('--monitor_mode', type=str, default='min', choices=['min', 'max'], help='Whether lower or higher metric is better')
parser.add_argument("--checkpoint_every_n_epoch", type=int, default=5, help="Number of the models to save")
parser.add_argument('--save_last', type=str2bool, default=True, help='Whether to save the last model')
parser.add_argument('--save_weights_only', type=str2bool, default=False, help='Whether to save only the model weights')
parser.add_argument('--log_every_n_steps', type=int, default=20, help='Log every N steps, to prevent model checkpoint does not save we must set this value lesser then or equal to total batch per gpu')
parser.add_argument('--save_SAIE_wight', type=str2bool, default=True, help='Whether to save SAIE weights')
# ----------------------------------------- Data Infomation ----------------------------------------------
# Data paths
'''
observation_datasets": Make it as [] if you want to close obervation during training.
observation_datasets_P300ERP": Make it as [] if you want to close obervation during training.
'''

parser.add_argument('--data_name', type=str, default='A', help='Dataset name')
parser.add_argument('--Dataset_style', type=str, default='BCICP300', choices=['Emotiv', 'MOABB','BCICP300','TUAB'], help="Path to EMOTIV EEG data file")
parser.add_argument('--emotiv_fusion_mode', type=str, default='multi_source_weighted', choices=['random_concat', 'multi_source_weighted', 'multi_source_truncated'], help='Fusion mode for Emotiv datasets')
parser.add_argument('--observation_datasets', nargs='+', default=["Attention", "STEW", "Crowdsource"], choices = ["Alpha", "Attention", "STEW", "Crowdsourced"],help='Make it as [] if you want to close obervation')
parser.add_argument('--observation_datasets_P300ERP', nargs='+', default=[],  choices = ["A", "B", "C"], help='Make it as [] if you want to close obervation')


# EEG Data Path
parser.add_argument('--Emotiv_path', type=str, default='/data/EEGMS/Processed/filter_zscore/Emotiv', help='Path to EMOTIV EEG data file')
parser.add_argument('--MOABB_path', type=str, default='/data/MOABB/Processed_filter_zscore', help='Path to PhysionetMI EEG data file')
parser.add_argument('--BCICP300_path', type=str, default='/data/EEGMS/Processed/nofilter_nozscore/BCICP300/Aggregation', help='Path to BCICP300 EEG data file')
parser.add_argument('--TUAB_path', type=str, default='/data/EEGMS/Processed/TUAB_16', help='Path to TUAB EEG data file')
parser.add_argument('--TU_sequen', type=int, default=12800)
parser
# EEG Data Montage
parser.add_argument("--Emotiv_coord_path", type=str, default="fm/EEGMS/Montage/Emotiv.xlsx", help="Path to EMOTIV EEG channel coordinate file")
parser.add_argument('--TUAB_coord_path', type=str, default='fm/EEGMS/Montage/TUAB_16.xlsx', help='Path to TUAB EEG channel coordinate file')
parser.add_argument('--BCICP300_coord_path', type=str, default='fm/EEGMS/Montage/BCICP300_64.xlsx', help='Path to BCICP300 coordinate file')
parser.add_argument('--PhysionetMI_coord_path', type=str, default='fm/EEGMS/Montage/PhysionetMI_64.xlsx', help='')
parser.add_argument('--BNCI2014_001_coord_path', type=str, default='fm/EEGMS/Montage/BNCI2014_001_22.xlsx', help='Path to BNCI2014_001 coordinate file')
parser.add_argument('--Emotiv_PhysionetMI_coord_path', type=str, default='fm/EEGMS/Montage/Emotiv_PhysionetMI.xlsx', help='')
parser.add_argument('--Standard_coord_path', type=str, default='fm/EEGMS/Montage/standard_1020_positions.xlsx', help='Path to 94 EEG channels form Standard 10-20 system acqurired from MNE')
# Inference
parser.add_argument('--inference', type=str2bool, default=False, help='If true, load model from checkpoint for inference')
parser.add_argument('--fine_numlayers_totrain', type=int, default=500, help='use for fine tuning. set a large number if you do not know the exactly layers of model')
parser.add_argument('--finetune_epochs', type=int, default=5, help='Number of epochs for fine-tuning')
parser.add_argument('--finetune_lr', type=int, default=1e-4)
# Oberservation metric 
parser.add_argument('--class_weights', nargs='+', type=float, default=[1.0, 1.0], help='List of class weights, e.g., 1.0 5.0')
parser.add_argument('--observation_logger', type=str2bool, default=False, help='Whether to logger observation log.')
parser.add_argument('--observation_interval', type=int, default=5, help='Run linear probing every N epochs')
parser.add_argument('--test_binaryF1', type=str2bool, default=True, help='Whether to calculate test_binaryF1 during training for observation.')
parser.add_argument('--target_layer', type=str, default='pool3', help= 'Which layer used as representation for downstream task')
parser.add_argument('--test_linear_probe', type=str2bool, default=True, help='Whether to calculate performance of linear probing during training for observation.')
parser.add_argument('--test_nonlinear_probe', type=str2bool, default=False, help='Whether to calculate performance of nonlinear probing during training for observation.')
parser.add_argument('--test_fine_tuning', type=str2bool, default=False, help='Whether to calculate performance of fine_tuning during training for observation.')
# --------------------------------------------Corpus I/O Parameters-----------------------------------------------------
parser.add_argument('--use_corpus', type=str2bool, default=True, help='Axon Corpus Default setting test')
parser.add_argument('--corpus_data_path', default='/data/sessions-small', choices={'/data/sessions-small', '/data/sessions-tiny'})
parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio for corpus pretraining')
parser.add_argument('--test_ratio', type=float, default=0.0, help='Test set ratio for corpus pretraining')
parser.add_argument('--corpus_subset', default=5, choices={1,2, 3, 4, 5}, help="Corpus subset")
parser.add_argument('--hertz', type=int, default=128, help='')
parser.add_argument('--channels', type=int, default=14, help='')
parser.add_argument('--seconds', type=int, default=2, help='')
parser.add_argument('--stride', type=int, default=64, help='')
# ----------------------------------------- Model Hyperparameters ----------------------------------------------
parser.add_argument('--d_state', type=int, default=16)
parser.add_argument('--d_conv', type=int, default=2)
parser.add_argument('--expand', type=int, default=8)
# ----------------------------------------- GCP Configuration --------------------------------------------------------
parser.add_argument('--corpus_input_path', type=str, default='/data/binary_files_by_subjects', help='Corpus input path')
parser.add_argument('--pretrain_dataset_percent', type=float, default=1.0, help='For testing, we can use a smaller subset of the data')
parser.add_argument('--predefined_file_list', type=str, default='', help='Predefined file list for training, if any')
# ----------------------------------------- Main ----------------------------------------------
args = parser.parse_args()
args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
config = SimpleNamespace(**vars(args))


if __name__ == "__main__":
    # ensure model initialization parameters across all ranks are the same
    seed_everything(config.seed, workers=True)
    # data = data_loader(config)
    data = fused_emotiv_loader(config)
    # data = fused_emotiv_physionet_loader(config)
    if config.Training_mode == 'Reconstruction':
        lightning_model, base_model, model_location = Reconstruction(config, data)
    else:
        config.inference = True
        model_location = '...'
        model = '...'
        if config.Training_mode == 'Linear_Probe': 
            result = run_linear_probe(model, config, data, probe_type='Linear', seed = 3047)
            print('results:', result)

        elif config.Training_mode == 'Finetune':
            result = run_downstream_finetune(model, config, data)
        else:
            raise ValueError("Downstream mode does not exist.")
