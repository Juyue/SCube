def hparams_handler(hparams):
    if not hasattr(hparams, 'ema'):
        hparams.ema = False
    if not hasattr(hparams, 'use_ddim'):
            hparams.use_ddim = False
    if not hasattr(hparams, 'scale_by_std'):
        hparams.scale_by_std = False
    if not hasattr(hparams, 'scale_factor'):
        hparams.scale_factor = 1.0
    if not hasattr(hparams, 'num_inference_steps'):
        hparams.num_inference_steps = 1000
    if not hasattr(hparams, 'conditioning_key'):
        hparams.conditioning_key = "none"
    if not hasattr(hparams, 'log_image'):
        hparams.log_image = True

        # position embedding
    if not hasattr(hparams, 'use_pos_embed'):
        hparams.use_pos_embed = False
    if not hasattr(hparams, 'use_pos_embed_high'):
        hparams.use_pos_embed_high = False
    if not hasattr(hparams, 'use_pos_embed_world'):
        hparams.use_pos_embed_world = False
    if not hasattr(hparams, 'use_pos_embed_world_high'):
        hparams.use_pos_embed_world_high = False

    # setup diffusion condition
    if not hasattr(hparams, 'use_mask_cond'):
        hparams.use_mask_cond = False
    if not hasattr(hparams, 'use_point_cond'):
        hparams.use_point_cond = False
    if not hasattr(hparams, 'use_semantic_cond'):
        hparams.use_semantic_cond = False
    if not hasattr(hparams, 'use_normal_concat_cond'):
        hparams.use_normal_concat_cond = False 
            
    if not hasattr(hparams, 'use_single_scan_concat_cond'):
        hparams.use_single_scan_concat_cond = False
    if not hasattr(hparams, 'encode_single_scan_by_points'):
        hparams.encode_single_scan_by_points = False
            
    if not hasattr(hparams, 'use_class_cond'):
        hparams.use_class_cond = False
    if not hasattr(hparams, 'use_micro_cond'):
        hparams.use_micro_cond = False
    if not hasattr(hparams, 'use_text_cond'):
        hparams.use_text_cond = False
    if not hasattr(hparams, 'use_image_w_depth_cond'):
        hparams.use_image_w_depth_cond = False
    if not hasattr(hparams, 'use_image_lss_cond'):
        hparams.use_image_lss_cond = False
    if not hasattr(hparams, 'use_map_3d_cond'):
        hparams.use_map_3d_cond = False
    if not hasattr(hparams, 'use_box_3d_cond'):
        hparams.use_box_3d_cond = False

    # noise offset config
    if not hasattr(hparams, 'use_noise_offset'):
        hparams.use_noise_offset = False
            
    # classifier-free config
    if not hasattr(hparams, 'use_classifier_free'):
        hparams.use_classifier_free = False # text cond in not influenced by this flag
    if not hasattr(hparams, 'classifier_free_prob'):
        hparams.classifier_free_prob = 0.1 # prob to drop the label
            
    # finetune config
    if not hasattr(hparams, 'pretrained_model_name_or_path'):
        hparams.pretrained_model_name_or_path = None
    if not hasattr(hparams, 'ignore_mismatched_size'):
        hparams.ignore_mismatched_size = False

    # vae config
    if not hasattr(hparams, 'finetune_vae_decoder'):
        hparams.finetune_vae_decoder = False

    return hparams