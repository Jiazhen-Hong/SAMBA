# Checkpoints

Download from Hugging Face: [SAMBA-2s](https://huggingface.co/Dcode-AI/SAMBA-2s) · [SAMBA-100s](https://huggingface.co/Dcode-AI/SAMBA-100s).

SAMBA-2s and SAMBA-100s were retrained after the original internship checkpoints
were lost and the data pipeline was reorganized. They are evaluated separately
from the paper's SAMBA-E and SAMBA-T.

| | SAMBA-2s | SAMBA-100s |
|---|---|---|
| Window | 256 samples at 128 Hz | 12,800 samples at 128 Hz |
| Sources | Alpha, Attention, Crowdsource, DriverDistraction, STEW, HBN resting state, Muse sleep onset, DICE eyes closed | TUAB |
| Input channels in pretraining | 4, 14, 19, 129 | 16 bipolar channels |
| Reconstruction | Input coordinates | Fixed 16-channel TUAB montage |

Pretraining uses per-window, per-channel absolute 95th-percentile normalization.
Inputs need valid electrode coordinates in the same units and coordinate frame
as the checkpoint's target positions. The window suffix does not limit encoder
input length. The SAMBA-100s reconstruction decoder retains its 16-channel
output even when the encoder receives a different montage.

Each model directory contains `config.json` and `model.safetensors`. Use the validation-best
weights for evaluation; optimizer state and TensorBoard logs stay in the
training repository. Load with `samba_eeg.SambaModel.from_pretrained` after
downloading the model directory. Transformers `AutoModel` is not supported.
Released Mamba-2 weights require CUDA for inference.

## NeuralBench

SAMBA-100s, frozen encoder, `linear_probe_mean`, seeds 33/34/35. Values are
mean ± sample standard deviation. Balanced accuracy uses `test/bal_acc`.

| Task | Balanced accuracy (%) | AUROC |
|---|---:|---:|
| Pathology | 71.44 ± 0.03 | 0.7914 ± 0.0003 |
| Depression | 79.81 ± 0.56 | 0.8127 ± 0.0262 |
| Schizophrenia | 62.51 ± 0.18 | 0.6553 ± 0.0011 |

Pathology shares the TUAB data domain with pretraining. The other two tasks
evaluate transfer to different datasets. These scores use NeuralBench's
protocol and are not directly comparable with the paper's fine-tuning results.
SAMBA-2s results are pending.
