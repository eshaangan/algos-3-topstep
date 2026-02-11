"""
GCP orchestrator for distributed grid search.

Spawns VMs, distributes experiments, monitors progress.
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_grid_config(config_path: Path) -> Dict:
    """Load grid search configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_phase1_configs(grid_config: Dict, num_samples: int = None) -> List[Dict]:
    """
    Generate Phase 1 experiment configurations using Latin hypercube sampling.
    
    Args:
        grid_config: Grid configuration dictionary
    
    Returns:
        List of experiment configs
    """
    phase1 = grid_config['phase1']
    if num_samples is None:
        num_samples = phase1['num_samples']

    logger.info(f"Generating {num_samples} Phase 1 configurations...")
    
    # Define the search space dimensions
    model_names = list(phase1['model_configs'].keys())
    feature_set_names = list(phase1['feature_sets'].keys())
    training_windows = phase1['training_windows']
    labeling_barriers = phase1['labeling_barriers']
    sample_weights = phase1['sample_weights']
    calibration_methods = phase1['calibration_methods']
    
    # For simplicity, use random sampling instead of true Latin hypercube
    # (True LH would require scipy, which may not be available on VM)
    np.random.seed(42)
    
    configs = []
    for i in range(num_samples):
        exp_id = f"phase1_exp_{i+1:04d}"
        
        # Sample from each dimension
        model_name = np.random.choice(model_names)
        feature_set_name = np.random.choice(feature_set_names)
        training_window = np.random.choice(training_windows)
        labeling = np.random.choice(labeling_barriers)
        sample_weight = np.random.choice(sample_weights)
        calibration = np.random.choice(calibration_methods)
        
        config = {
            'exp_id': exp_id,
            'phase': 1,
            'model_name': model_name,
            'model_params': phase1['model_configs'][model_name],
            'feature_set_name': feature_set_name,
            'feature_set': phase1['feature_sets'][feature_set_name],
            'training_window_months': training_window,
            'labeling': labeling,
            'sample_weight': sample_weight,
            'calibration': calibration
        }
        
        configs.append(config)
    
    logger.info(f"Generated {len(configs)} Phase 1 configs")
    return configs


def generate_phase2_configs(grid_config: Dict, top_configs: List[Dict]) -> List[Dict]:
    """
    Generate Phase 2 configurations by creating neighborhoods around top Phase 1 configs.
    
    Args:
        grid_config: Grid configuration dictionary
        top_configs: Top configurations from Phase 1
    
    Returns:
        List of experiment configs
    """
    phase2 = grid_config['phase2']
    deltas = phase2['hyperparameter_deltas']
    
    logger.info(f"Generating Phase 2 configs from {len(top_configs)} base configs...")
    
    configs = []
    exp_num = 1
    
    for base_config in top_configs:
        base_params = base_config['model_params']
        
        # Create variations
        for n_est_delta in deltas['n_estimators']:
            for max_depth_delta in deltas['max_depth']:
                for lr in deltas['learning_rate']:
                    for ff in deltas['feature_fraction']:
                        for bf in deltas['bagging_fraction']:
                            # Create new config
                            new_params = base_params.copy()
                            new_params['n_estimators'] = max(10, base_params['n_estimators'] + n_est_delta)
                            new_params['max_depth'] = max(2, base_params['max_depth'] + max_depth_delta)
                            new_params['learning_rate'] = lr
                            new_params['feature_fraction'] = ff
                            new_params['bagging_fraction'] = bf
                            
                            exp_id = f"phase2_exp_{exp_num:04d}"
                            exp_num += 1
                            
                            config = {
                                'exp_id': exp_id,
                                'phase': 2,
                                'base_exp_id': base_config['exp_id'],
                                'model_name': f"{base_config['model_name']}_tuned",
                                'model_params': new_params,
                                'feature_set_name': base_config['feature_set_name'],
                                'feature_set': base_config['feature_set'],
                                'training_window_months': base_config['training_window_months'],
                                'labeling': base_config['labeling'],
                                'sample_weight': base_config['sample_weight'],
                                'calibration': base_config['calibration']
                            }
                            
                            configs.append(config)
    
    logger.info(f"Generated {len(configs)} Phase 2 configs")
    return configs


def split_into_batches(configs: List[Dict], num_batches: int) -> List[List[Dict]]:
    """Split experiment configs into batches for parallel execution."""
    batch_size = len(configs) // num_batches
    remainder = len(configs) % num_batches
    
    batches = []
    start = 0
    for i in range(num_batches):
        # Add 1 extra to first 'remainder' batches to distribute evenly
        size = batch_size + (1 if i < remainder else 0)
        batches.append(configs[start:start + size])
        start += size
    
    return batches


def upload_to_gcs(local_path: Path, gcs_path: str):
    """Upload file to Google Cloud Storage."""
    cmd = ['gsutil', 'cp', str(local_path), gcs_path]
    subprocess.run(cmd, check=True)
    logger.info(f"Uploaded {local_path} to {gcs_path}")


def download_from_gcs(gcs_path: str, local_path: Path):
    """Download file from Google Cloud Storage."""
    cmd = ['gsutil', 'cp', gcs_path, str(local_path)]
    subprocess.run(cmd, check=True)
    logger.info(f"Downloaded {gcs_path} to {local_path}")


def create_vm(
    vm_name: str,
    zone: str,
    machine_type: str,
    disk_size_gb: int,
    startup_script_path: Path,
    batch_file_gcs: str,
    phase: int
):
    """Create a GCP compute VM."""
    cmd = [
        'gcloud', 'compute', 'instances', 'create', vm_name,
        f'--zone={zone}',
        f'--machine-type={machine_type}',
        '--image-family=ubuntu-2004-lts',
        '--image-project=ubuntu-os-cloud',
        f'--boot-disk-size={disk_size_gb}GB',
        '--scopes=storage-full,compute-rw',
        f'--metadata-from-file=startup-script={startup_script_path}',
        f'--metadata=batch-file={batch_file_gcs},phase={phase}'
    ]
    
    logger.info(f"Creating VM: {vm_name}")
    subprocess.run(cmd, check=True)


def delete_vm(vm_name: str, zone: str):
    """Delete a GCP compute VM."""
    cmd = [
        'gcloud', 'compute', 'instances', 'delete', vm_name,
        f'--zone={zone}',
        '--quiet'
    ]
    
    logger.info(f"Deleting VM: {vm_name}")
    subprocess.run(cmd, check=True, capture_output=True)


def list_gcs_files(gcs_prefix: str) -> List[str]:
    """List files in GCS bucket with given prefix."""
    cmd = ['gsutil', 'ls', gcs_prefix]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().split('\n') if result.stdout.strip() else []


def monitor_completion(
    gcs_results_prefix: str,
    total_experiments: int,
    check_interval_seconds: int = 60
) -> int:
    """
    Monitor experiment completion by counting result files in GCS.
    
    Returns:
        Number of completed experiments
    """
    completed_files = list_gcs_files(gcs_results_prefix)
    completed = len([f for f in completed_files if f.endswith('.json')])
    
    logger.info(f"Progress: {completed}/{total_experiments} experiments completed ({100*completed/total_experiments:.1f}%)")
    
    return completed


def run_phase(
    phase: int,
    grid_config: Dict,
    num_vms: int,
    base_configs: List[Dict] = None
):
    """
    Run a full phase of grid search on GCP.
    
    Args:
        phase: Phase number (1, 2, or 3)
        grid_config: Grid configuration dictionary
        num_vms: Number of VMs to spawn
        base_configs: For Phase 2/3, the top configs from previous phase
    """
    logger.info(f"=== Starting Phase {phase} ===")
    
    # Generate experiment configs
    if phase == 1:
        configs = generate_phase1_configs(grid_config)
    elif phase == 2:
        if not base_configs:
            raise ValueError("Phase 2 requires base_configs from Phase 1")
        configs = generate_phase2_configs(grid_config, base_configs)
    else:
        raise NotImplementedError(f"Phase {phase} not implemented yet")
    
    # Split into batches
    batches = split_into_batches(configs, num_vms)
    
    # Upload batches to GCS
    gcp_config = grid_config['gcp']
    bucket = gcp_config['bucket']
    
    batch_files_gcs = []
    for i, batch in enumerate(batches):
        batch_file = Path(f'/tmp/phase{phase}_batch_{i}.json')
        with open(batch_file, 'w') as f:
            json.dump(batch, f, indent=2)
        
        gcs_path = f"{bucket}/experiment-batches/phase{phase}/batch_{i}.json"
        upload_to_gcs(batch_file, gcs_path)
        batch_files_gcs.append(gcs_path)
    
    # Create VMs
    zone = gcp_config['zone']
    machine_type = gcp_config['machine_type']
    disk_size = gcp_config['disk_size_gb']
    startup_script = Path(__file__).parent / 'gcp_startup.sh'
    
    vm_names = []
    for i in range(num_vms):
        vm_name = f"grid-worker-phase{phase}-{i}"
        vm_names.append(vm_name)
        
        try:
            create_vm(
                vm_name,
                zone,
                machine_type,
                disk_size,
                startup_script,
                batch_files_gcs[i],
                phase
            )
        except Exception as e:
            logger.error(f"Failed to create VM {vm_name}: {e}")
    
    # Monitor completion
    gcs_results_prefix = f"{bucket}/experiment-results/phase{phase}/"
    total_experiments = len(configs)
    
    logger.info(f"Monitoring {total_experiments} experiments across {num_vms} VMs...")
    
    start_time = time.time()
    while True:
        completed = monitor_completion(gcs_results_prefix, total_experiments)
        
        if completed >= total_experiments:
            logger.info("All experiments completed!")
            break
        
        # Check timeout (max 4 hours per phase)
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours > 4:
            logger.warning(f"Phase timeout reached ({elapsed_hours:.1f} hours)")
            break
        
        time.sleep(60)  # Check every minute
    
    # Cleanup VMs
    logger.info("Cleaning up VMs...")
    for vm_name in vm_names:
        try:
            delete_vm(vm_name, zone)
        except Exception as e:
            logger.error(f"Failed to delete VM {vm_name}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Phase {phase} completed in {elapsed/3600:.2f} hours")


def main():
    parser = argparse.ArgumentParser(description="GCP Grid Search Orchestrator")
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3], help='Phase number')
    parser.add_argument('--num-vms', type=int, default=10, help='Number of VMs to spawn')
    parser.add_argument('--config', type=str, default='grid_config.yaml', help='Grid config file')
    parser.add_argument('--base-configs', type=str, help='JSON file with base configs (Phase 2/3)')
    
    args = parser.parse_args()
    
    # Load grid config
    config_path = Path(__file__).parent / args.config
    grid_config = load_grid_config(config_path)
    
    # Load base configs if provided
    base_configs = None
    if args.base_configs:
        with open(args.base_configs, 'r') as f:
            base_configs = json.load(f)
    
    # Run phase
    run_phase(args.phase, grid_config, args.num_vms, base_configs)


if __name__ == '__main__':
    main()
