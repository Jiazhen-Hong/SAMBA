import torch

def load_compatible_weights(model, ckpt_path, device, logger=None, load_weights=True):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict_key = None
    if 'model_state_dict' in checkpoint:
        state_dict_key = 'model_state_dict'
    elif 'state_dict' in checkpoint:
        state_dict_key = 'state_dict'
    
    pretrained_dict = checkpoint[state_dict_key] if state_dict_key else checkpoint
    pretrained_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in pretrained_dict.items()}

    model_dict = model.state_dict()
    compatible_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    skipped_keys = [k for k in model_dict if k not in compatible_dict]

    if logger:
        logger.info(f"[Check] Compatible weights: {len(compatible_dict)} / {len(model_dict)}")
        logger.info(f"[Check] Skipped keys: {skipped_keys}")
    else:
        print(f"[Check] Compatible weights: {len(compatible_dict)} / {len(model_dict)}")
        print(f"[Check] Skipped keys: {skipped_keys}")

    if load_weights:
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        if logger:
            logger.info("[Check] Loaded compatible weights into model.")