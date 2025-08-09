import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.functional import cross_entropy
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import pandas as pd
import seaborn as sns
from scipy import stats # Import for statistical tests
from tqdm import tqdm # Import for progress bars

print("Starting Adversarial Attack & Defense Script...", flush=True)
print("---", flush=True)

# true to run a brief adversarial fine-tuning session on the black-box model
PERFORM_ADVERSARIAL_TRAINING = True
# a new model 'checkpoint_best_robust.pth' will be created and evaluated.

image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
gt_base_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations"
output_base_dir = "/sharedscratch/an252/cancerdetectiondataset/brats_attacks_multi_run_v2"
checkpoint_path_black_box = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"
checkpoint_path_surrogate = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
checkpoint_path_robust = checkpoint_path_black_box.replace('checkpoint_best.pth', 'checkpoint_best_robust.pth')
plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"
dataset_json_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

try:
    dataset_json_content = load_json(dataset_json_path)
    plans_json_content = load_json(plans_path)
except FileNotFoundError as e:
    print(f"Error loading JSON files: {e}. Please verify paths.", flush=True)
    exit()

print("Initializing nnUNetTrainer and loading models...", flush=True)
models = {}
try:
    trainer_black_box = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3, dataset_json=dataset_json_content, device=device)
    trainer_black_box.initialize()
    # add weights_only=False
    checkpoint_black_box = torch.load(checkpoint_path_black_box, map_location=device, weights_only=False)
    model_black_box = trainer_black_box.network.eval()
    model_black_box.load_state_dict(checkpoint_black_box['network_weights'])
    models['black_box_fold3'] = model_black_box
    print(f"Black-Box Model (Fold 3) loaded from {checkpoint_path_black_box}", flush=True)

    # surrogate Model (Fold 0)
    trainer_surrogate = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=0, dataset_json=dataset_json_content, device=device)
    trainer_surrogate.initialize()
    checkpoint_surrogate = torch.load(checkpoint_path_surrogate, map_location=device, weights_only=False)
    model_surrogate = trainer_surrogate.network.eval()
    model_surrogate.load_state_dict(checkpoint_surrogate['network_weights'])
    models['surrogate_fold0'] = model_surrogate
    print(f"Surrogate Model (Fold 0) loaded from {checkpoint_path_surrogate}", flush=True)

except Exception as e:
    print(f"FATAL ERROR: Could not initialize or load models: {e}", flush=True)
    exit()

# helper function to extract logits from nnUNet output 
def _get_logits_from_nnunet_output(output):
    if isinstance(output, (list, tuple)):
        return _get_logits_from_nnunet_output(output[0])
    return output

# custom Preprocessing Function (remains the same) 
def preprocess_single_image_for_nnunet(raw_image_data_np, raw_image_properties, plans_dict, device):
    configuration = "3d_fullres"
    current_config = plans_dict['configurations'][configuration]
    target_spacing_plans = np.array(current_config['spacing'])
    intensity_properties = plans_dict['foreground_intensity_properties_per_channel']
    original_spacing_xyz = np.array(raw_image_properties['spacing'])
    sitk_stuff = raw_image_properties['sitk_stuff']
    original_origin_xyz = sitk_stuff['origin']
    original_direction_matrix = sitk_stuff['direction']
    sitk_images = []
    for c in range(raw_image_data_np.shape[0]):
        img_channel_np_xyz = raw_image_data_np[c].transpose(2, 1, 0)
        sitk_img = sitk.GetImageFromArray(img_channel_np_xyz)
        sitk_img.SetSpacing(original_spacing_xyz)
        sitk_img.SetOrigin(original_origin_xyz)
        sitk_img.SetDirection(original_direction_matrix)
        sitk_images.append(sitk_img)
    current_shape_xyz = np.array(sitk_images[0].GetSize())
    target_spacing_xyz = target_spacing_plans[::-1]
    new_size_xyz = np.round(current_shape_xyz * original_spacing_xyz / target_spacing_xyz).astype(int)
    divisible_by = 2**(len(current_config['architecture']['arch_kwargs']['strides']) - 1)
    padded_size_xyz = np.ceil(new_size_xyz / divisible_by).astype(int) * divisible_by
    resampled_data_channels_np_dhw = []
    for sitk_img in sitk_images:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_img)
        resampler.SetOutputSpacing(target_spacing_xyz)
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetSize(padded_size_xyz.tolist())
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampled_img_sitk = resampler.Execute(sitk_img)
        resampled_data_channels_np_dhw.append(sitk.GetArrayFromImage(resampled_img_sitk).transpose(2, 1, 0))
    preprocessed_image_np = np.stack(resampled_data_channels_np_dhw, axis=0)
    normalized_image_channels = []
    for c in range(preprocessed_image_np.shape[0]):
        channel_data = preprocessed_image_np[c]
        mean_val = intensity_properties[str(c)]['mean']
        std_val = intensity_properties[str(c)]['std'] + 1e-6
        normalized_channel = (channel_data - mean_val) / std_val
        normalized_image_channels.append(normalized_channel)
    preprocessed_image_np = np.stack(normalized_image_channels, axis=0)
    img_tensor_normalized = torch.tensor(preprocessed_image_np, dtype=torch.float32).unsqueeze(0).to(device)
    return img_tensor_normalized

# dice Calculation (remains the same) 
def dice(gt_seg, pred_seg, labels_to_include):
    if not isinstance(labels_to_include, (list, tuple)):
        labels_to_include = [labels_to_include]
    gt_bin = torch.zeros_like(gt_seg, dtype=torch.bool)
    pred_bin = torch.zeros_like(pred_seg, dtype=torch.bool)
    for label in labels_to_include:
        gt_bin = gt_bin | (gt_seg == label)
        pred_bin = pred_bin | (pred_seg == label)
    gt_bin_np = gt_bin.cpu().numpy()
    pred_bin_np = pred_bin.cpu().numpy()
    intersection = (gt_bin_np & pred_bin_np).sum()
    union = gt_bin_np.sum() + pred_bin_np.sum()
    if union == 0:
        return 1.0
    return 2.0 * intersection / (union + 1e-6)

# define tumor regions
TUMOR_REGIONS = {"ET": [3], "ED": [2], "NCR_NET": [1], "WT": [1, 2, 3], "TC": [1, 3]}

# aversarial fine-tuning Function 
def adversarial_fine_tuning(model_to_train, patient_ids_for_tuning, plans, attack_params, device, save_path):
    print("\n--- Starting Adversarial Fine-Tuning ---", flush=True)
    model_to_train.train() # model to training mode
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=1e-5)
    image_loader = SimpleITKIO()
    
    for epoch in range(2): # fine-tune for 2 epochs
        print(f"Fine-Tuning Epoch {epoch + 1}/2", flush=True)
        for patient_id in tqdm(patient_ids_for_tuning, desc=f"Epoch {epoch+1}"):
            try:
                # load and preprocess data
                full_image_paths = [os.path.join(image_dir, f"{patient_id}_{i:04d}.nii.gz") for i in range(4)]
                image_data, image_props = image_loader.read_images(full_image_paths)
                gt_path = os.path.join(gt_base_dir, f"{patient_id}.nii.gz")
                gt_nifti = nib.load(gt_path)
                gt_full_volume_raw = gt_nifti.get_fdata()

                img_tensor = preprocess_single_image_for_nnunet(image_data, image_props, plans, device)

                # preprocess ground truth
                target_dims = img_tensor.shape[2:]
                gt_tensor = torch.tensor(gt_full_volume_raw.astype(np.int64)).to(device).unsqueeze(0).unsqueeze(0)
                gt_tensor = torch.nn.functional.interpolate(gt_tensor.float(), size=target_dims, mode='nearest-exact').long()
                
                # generate Adversarial using PGD ---
                # PGD is a strong attack, good for robust training
                adv_img_tensor = img_tensor.clone().detach()
                alpha = attack_params['alpha']
                epsilon = attack_params['epsilon']

                for _ in range(attack_params['num_iter']):
                    adv_img_tensor.requires_grad = True
                    outputs = _get_logits_from_nnunet_output(model_to_train(adv_img_tensor))
                    loss = cross_entropy(outputs, gt_tensor.squeeze(0))
                    
                    model_to_train.zero_grad()
                    loss.backward()
                    
                    # update image
                    perturbation = alpha * adv_img_tensor.grad.sign()
                    adv_img_tensor = adv_img_tensor.detach() + perturbation
                    # project perturbation to epsilon ball
                    total_perturbation = torch.clamp(adv_img_tensor - img_tensor, -epsilon, epsilon)
                    adv_img_tensor = img_tensor + total_perturbation
                    adv_img_tensor = torch.clamp(adv_img_tensor, -5, 5) # Clamp to valid range

                # training step on Adversarial Example ---
                optimizer.zero_grad()
                adv_outputs = _get_logits_from_nnunet_output(model_to_train(adv_img_tensor))
                training_loss = cross_entropy(adv_outputs, gt_tensor.squeeze(0))
                training_loss.backward()
                optimizer.step()
                
            except FileNotFoundError:
                print(f"Skipping {patient_id} for tuning, file not found.", flush=True)
                continue
            except Exception as e:
                print(f"Error during fine-tuning on {patient_id}: {e}", flush=True)
                continue
    
    # save the fine-tuned model weights
    model_to_train.eval() 
    torch.save({'network_weights': model_to_train.state_dict()}, save_path)
    print(f"--- Adversarial Fine-Tuning Complete. Robust model saved to:\n{save_path} ---", flush=True)
    return model_to_train

# refactored Attack Evaluation Function ---
def run_attack_evaluation(patient_id, slice_idx, attack_config, models_dict, device, output_base_dir,
                          image_dir, gt_base_dir, plans_json_content, tumor_regions_dict):

    attack_type = attack_config['name']
    epsilon = attack_config['epsilon']
    
    # determine target and gradient models from config
    target_model = models_dict[attack_config['target_model_name']]
    gradient_model = models_dict.get(attack_config['grad_model_name']) # .get for noise case

    print(f"\n--- Running Attack: {attack_type} | Patient: {patient_id} | Epsilon: {epsilon:.3f} ---", flush=True)
    
    attack_output_dir = os.path.join(output_base_dir, attack_type, f"{patient_id}_eps{epsilon:.3f}")
    os.makedirs(attack_output_dir, exist_ok=True)

    result_data = {
        'patient_id': patient_id, 'slice_idx': slice_idx, 'epsilon': epsilon,
        'attack_type': attack_type, 'target_model': attack_config['target_model_name'],
        'status': 'Success'
    }

    try:
        # load & Preprocess Data (image and ground truth)
        image_loader = SimpleITKIO()
        full_image_paths = [os.path.join(image_dir, f"{patient_id}_{i:04d}.nii.gz") for i in range(4)]
        loaded_image_data, loaded_image_properties = image_loader.read_images(full_image_paths)
        
        gt_path = os.path.join(gt_base_dir, f"{patient_id}.nii.gz")
        gt_nifti = nib.load(gt_path)
        gt_full_volume_raw = gt_nifti.get_fdata()

        img_tensor_normalized = preprocess_single_image_for_nnunet(
            loaded_image_data, loaded_image_properties, plans_json_content, device
        )

        target_D, target_H, target_W = img_tensor_normalized.shape[2:]
        gt_tensor = torch.tensor(gt_full_volume_raw.astype(np.int64)).to(device).unsqueeze(0).unsqueeze(0)
        gt_preprocessed = torch.nn.functional.interpolate(gt_tensor.float(),
                                                         size=(target_D, target_H, target_W),
                                                         mode='nearest-exact').squeeze(0).long()
        gt_slice_tensor = gt_preprocessed.squeeze(0)[slice_idx, :, :].clone()

        # original prediction (on target model) 
        with torch.no_grad():
            output_logits_orig = _get_logits_from_nnunet_output(target_model(img_tensor_normalized))
            pred_slice = output_logits_orig.argmax(dim=1).squeeze(0)[slice_idx, :, :]

        # generate adversarial example
        adv_image_tensor = None
        data_grad = None # for heatmap visualization
        
        img_tensor_for_attack = img_tensor_normalized.clone().detach()

        if attack_config['type'] == 'fgsm':
            img_tensor_for_attack.requires_grad = True
            output_logits_grad = _get_logits_from_nnunet_output(gradient_model(img_tensor_for_attack))
            loss = cross_entropy(output_logits_grad, gt_preprocessed)
            gradient_model.zero_grad()
            loss.backward()
            data_grad = img_tensor_for_attack.grad.data
            adv_image_tensor = img_tensor_normalized + epsilon * data_grad.sign()

        elif attack_config['type'] == 'pgd':
            adv_image_tensor = img_tensor_for_attack.clone()
            alpha = attack_config['alpha']
            for i in range(attack_config['num_iter']):
                adv_image_tensor.requires_grad = True
                output_logits_grad = _get_logits_from_nnunet_output(gradient_model(adv_image_tensor))
                loss = cross_entropy(output_logits_grad, gt_preprocessed)
                gradient_model.zero_grad()
                loss.backward()
                # store gradient from the first iteration for visualization
                if i == 0:
                    data_grad = adv_image_tensor.grad.data.clone()
                
                perturbation = alpha * adv_image_tensor.grad.sign()
                adv_image_tensor = adv_image_tensor.detach() + perturbation
                total_perturbation = torch.clamp(adv_image_tensor - img_tensor_normalized, -epsilon, epsilon)
                adv_image_tensor = img_tensor_normalized + total_perturbation
        
        elif attack_config['type'] == 'gaussian_noise':
            noise = torch.randn_like(img_tensor_normalized) * epsilon
            adv_image_tensor = img_tensor_normalized + noise

        # clamp final adversarial image
        adv_image_tensor = torch.clamp(adv_image_tensor, -5, 5)

        # prediction (on target model)
        with torch.no_grad():
            adv_output_logits = _get_logits_from_nnunet_output(target_model(adv_image_tensor))
            adv_pred_slice = adv_output_logits.argmax(dim=1).squeeze(0)[slice_idx, :, :]

        # dice calculation & saving 
        for region_name, labels in tumor_regions_dict.items():
            d_orig = dice(gt_slice_tensor, pred_slice, labels)
            d_adv = dice(gt_slice_tensor, adv_pred_slice, labels)
            result_data[f'dice_orig_{region_name}'] = d_orig
            result_data[f'dice_adv_{region_name}'] = d_adv
        
        # save visualizations 
        fig, axs = plt.subplots(1, 4, figsize=(20, 6))
        
        # clean image
        original_slice_np = img_tensor_normalized[0, 1, slice_idx, :, :].cpu().numpy()
        axs[0].imshow(original_slice_np, cmap='gray')
        axs[0].set_title(f"Clean Image\nET Dice: {result_data['dice_orig_ET']:.3f}")
        axs[0].axis('off')

        #gradient map heatmap
        if data_grad is not None:
            grad_slice_np = data_grad[0, 1, slice_idx, :, :].cpu().abs().numpy()
            axs[1].imshow(grad_slice_np, cmap='hot')
            axs[1].set_title("Gradient Map (Sensitivity)")
        else:
            axs[1].text(0.5, 0.5, 'N/A for\nGaussian Noise', ha='center', va='center', fontsize=12)
            axs[1].set_title("Gradient Map")
        axs[1].axis('off')

        # perturbation
        perturbation_slice_np = (adv_image_tensor - img_tensor_normalized)[0, 1, slice_idx, :, :].cpu().numpy()
        max_abs_val = np.max(np.abs(perturbation_slice_np))
        im = axs[2].imshow(perturbation_slice_np, cmap='RdBu_r', vmin=-max_abs_val-1e-6, vmax=max_abs_val+1e-6)
        axs[2].set_title("Added Perturbation")
        axs[2].axis('off')
        fig.colorbar(im, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)

        # adversarial Image
        adv_slice_np = adv_image_tensor[0, 1, slice_idx, :, :].cpu().numpy()
        axs[3].imshow(adv_slice_np, cmap='gray')
        axs[3].set_title(f"Adversarial Image\nET Dice: {result_data['dice_adv_ET']:.3f}")
        axs[3].axis('off')

        plt.suptitle(f'Attack: {attack_type} | Patient: {patient_id} | Epsilon: {epsilon:.3f}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        overlay_save_path = os.path.join(attack_output_dir, f"{patient_id}_slice{slice_idx}.png")
        plt.savefig(overlay_save_path, dpi=150)
        plt.close()
        result_data['overlay_path'] = overlay_save_path

    except FileNotFoundError as e:
        result_data['status'] = f'Skipped: Missing file for {patient_id}'
    except Exception as e:
        result_data['status'] = f'Failed: {e}'
        print(f"ERROR processing {patient_id}: {e}", flush=True)

    return result_data

# main loop
if __name__ == "__main__":
    os.makedirs(output_base_dir, exist_ok=True)
    
    patient_cases_for_attack = [f"BraTS2021_{i:05d}" for i in range(5, 35)] # 30 patients for attack demo
    slices_to_attack = [75] # single slice for quicker demo
    
    # Adversarial Fine-Tuning 
    if PERFORM_ADVERSARIAL_TRAINING:
        tuning_patient_cases = [f"BraTS2021_{i:05d}" for i in range(15, 20)] # Use different patients for tuning
        tuning_attack_params = {'type': 'pgd', 'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 5}
        
        # Create a fresh copy of the model for training
        model_for_training = trainer_black_box.network
        model_for_training.load_state_dict(models['black_box_fold3'].state_dict())
        
        robust_model = adversarial_fine_tuning(
            model_to_train=model_for_training.to(device),
            patient_ids_for_tuning=tuning_patient_cases,
            plans=plans_json_content,
            attack_params=tuning_attack_params,
            device=device,
            save_path=checkpoint_path_robust
        )
        models['robust_model'] = robust_model.eval()
    
    # Define All Attack Configurations
    attack_configurations = []
    base_epsilons = [0.03, 0.06]
    
    # standard models
    target_models = ['black_box_fold3']
    # If we trained a robust model, add it to the list of targets to test its resilience
    if PERFORM_ADVERSARIAL_TRAINING and 'robust_model' in models:
        target_models.append('robust_model')

    for tm_name in target_models:
        # White-Box FGSM
        for eps in base_epsilons:
            attack_configurations.append({'name': f'WB-FGSM_on_{tm_name}', 'type': 'fgsm', 'epsilon': eps, 'target_model_name': tm_name, 'grad_model_name': tm_name})
        # White-Box PGD
        for eps in base_epsilons:
            attack_configurations.append({'name': f'WB-PGD_on_{tm_name}', 'type': 'pgd', 'epsilon': eps, 'alpha': eps/4, 'num_iter': 10, 'target_model_name': tm_name, 'grad_model_name': tm_name})
        # Black-Box FGSM (only on original model)
        if tm_name == 'black_box_fold3':
            for eps in base_epsilons:
                attack_configurations.append({'name': 'BB-FGSM', 'type': 'fgsm', 'epsilon': eps, 'target_model_name': 'black_box_fold3', 'grad_model_name': 'surrogate_fold0'})
        # Gaussian Noise Baseline
        for eps in [0.1, 0.2]: # Noise epsilon is std. dev, so needs to be larger
             attack_configurations.append({'name': f'Gaussian-Noise_on_{tm_name}', 'type': 'gaussian_noise', 'epsilon': eps, 'target_model_name': tm_name, 'grad_model_name': None})

    # run all attacks
    all_results = []
    total_runs = len(patient_cases_for_attack) * len(slices_to_attack) * len(attack_configurations)
    print(f"\nStarting evaluation of {total_runs} attack configurations...", flush=True)

    for patient_id in patient_cases_for_attack:
        for slice_idx in slices_to_attack:
            for config in attack_configurations:
                result = run_attack_evaluation(
                    patient_id, slice_idx, config, models, device, output_base_dir,
                    image_dir, gt_base_dir, plans_json_content, TUMOR_REGIONS
                )
                all_results.append(result)

    # save and analyze results
    print("\n--- All attacks completed. Saving and analyzing results. ---", flush=True)
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_base_dir, "adversarial_attack_summary.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Attack summary saved to:\n{results_csv_path}", flush=True)

    # visualization & statistical analysis ---
    OUTPUT_PLOTS_DIR = os.path.join(output_base_dir, "analysis_plots")
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    df_successful = results_df[results_df['status'] == 'Success'].copy()
    
    if not df_successful.empty:
        # convert dice columns to numeric
        for region in TUMOR_REGIONS.keys():
            df_successful[f'dice_orig_{region}'] = pd.to_numeric(df_successful[f'dice_orig_{region}'], errors='coerce')
            df_successful[f'dice_adv_{region}'] = pd.to_numeric(df_successful[f'dice_adv_{region}'], errors='coerce')
        df_successful.dropna(inplace=True)

        # bar Chart Visualization
        print("\nGenerating Bar Chart Visualization...", flush=True)
        melted_df = pd.melt(df_successful, 
                            id_vars=['attack_type', 'epsilon'], 
                            value_vars=[f'dice_orig_{r}' for r in TUMOR_REGIONS] + [f'dice_adv_{r}' for r in TUMOR_REGIONS],
                            var_name='Metric', value_name='Dice_Score')
        melted_df['Condition'] = melted_df['Metric'].apply(lambda x: 'Adversarial' if 'adv' in x else 'Original')
        melted_df['Region'] = melted_df['Metric'].apply(lambda x: x.split('_')[-1])
        
        g = sns.catplot(data=melted_df, x='Region', y='Dice_Score', hue='Condition', 
                        col='attack_type', col_wrap=3, kind='bar',
                        palette={'Original': 'skyblue', 'Adversarial': 'salmon'},
                        errorbar='sd', sharey=True)
        g.fig.suptitle('Average Dice Score: Clean vs. Adversarial by Attack Type', y=1.02)
        g.set_axis_labels("Tumor Region", "Average Dice Score")
        g.set_titles("{col_name}")
        g.set(ylim=(0, 1))
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        bar_chart_path = os.path.join(OUTPUT_PLOTS_DIR, "dice_scores_by_attack_type.png")
        plt.savefig(bar_chart_path, dpi=300)
        plt.close()
        print(f"Bar chart saved to: {bar_chart_path}", flush=True)
        
        # paired Statistical Tests
        print("\n--- Performing Paired Statistical Tests (Wilcoxon signed-rank) ---", flush=True)
        print("Comparing Original vs. Adversarial Dice scores for each attack type.")
        print("-" * 70)
        print(f"{'Attack Type':<25} {'Region':<10} {'P-Value':<15} {'Significance'}")
        print("-" * 70)
        
        for attack in df_successful['attack_type'].unique():
            attack_df = df_successful[df_successful['attack_type'] == attack]
            for region in TUMOR_REGIONS.keys():
                orig_scores = attack_df[f'dice_orig_{region}']
                adv_scores = attack_df[f'dice_adv_{region}']
                
                # Wilcoxon test requires at least a few data points
                if len(orig_scores) > 5 and not np.allclose(orig_scores, adv_scores):
                    try:
                        stat, p_value = stats.wilcoxon(orig_scores, adv_scores, zero_method='zsplit')
                        significance = '*** p < 0.001' if p_value < 0.001 else ('** p < 0.01' if p_value < 0.01 else ('* p < 0.05' if p_value < 0.05 else 'Not Significant'))
                        print(f"{attack:<25} {region:<10} {p_value:<15.5e} {significance}")
                    except ValueError as e:
                        # Happens if all differences are zero
                         print(f"{attack:<25} {region:<10} {'N/A':<15} (No difference in scores)")
                else:
                    print(f"{attack:<25} {region:<10} {'N/A':<15} (Not enough data or no change)")
        print("-" * 70)

    else:
        print("No successful attack results to analyze. Exiting analysis.", flush=True)

    print("\nScript finished successfully.", flush=True)