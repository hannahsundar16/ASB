import getpass
import os
import yaml
import argparse

def load_config(cfg_path):
    """Load configuration from YAML file."""
    with open(cfg_path, 'r') as file:
        return yaml.safe_load(file)

def validate_config(cfg):
    """Validate required configuration values."""
    required_fields = ['llms', 'attack_types']
    for field in required_fields:
        if field not in cfg or not cfg[field]:
            raise ValueError(f"Error: '{field}' must be specified in the configuration file.")

def get_api_key():
    """Get API key from user."""
    return getpass.getpass("Please enter your API key: ")

def determine_attacker_tools_path(attack_tool_type):
    """Determine attacker tools path based on attack tool type."""
    attacker_tools_paths = {
        'all': 'data/all_attack_tools.jsonl',
        'non-agg': 'data/all_attack_tools_non_aggressive.jsonl',
        'agg': 'data/all_attack_tools_aggressive.jsonl',
        'test': 'data/attack_tools_test.jsonl'
    }
    return attacker_tools_paths.get(attack_tool_type, 'data/all_attack_tools.jsonl')

def construct_base_command(llm_name, api_key, base_url, attack_type, backend, attacker_tools_path, log_file, suffix, database, write_db, read_db, defense_type):
    """Construct base command for main attacker script."""
    base_cmd = f'''nohup python main_attacker.py --llm_name "{llm_name}" --api_key {api_key} --base_url {base_url} --attack_type {attack_type} --use_backend {backend} --attacker_tools_path {attacker_tools_path} --res_file {log_file}_{suffix}.csv'''
    if database:
        base_cmd += f' --database {database}'
    if write_db:
        base_cmd += ' --write_db'
    if read_db:
        base_cmd += ' --read_db'
    if defense_type:
        base_cmd += f' --defense_type {defense_type}'
    return base_cmd

def add_injection_method_to_command(injection_method):
    """Add injection method to command."""
    injection_methods = {
        'direct_prompt_injection': ' --direct_prompt_injection',
        'memory_attack': ' --memory_attack',
        'observation_prompt_injection': ' --observation_prompt_injection',
        'clean': ' --clean',
        'mixed_attack': ' --direct_prompt_injection --observation_prompt_injection',
        'DPI_MP': ' --direct_prompt_injection',
        'OPI_MP': ' --observation_prompt_injection',
        'DPI_OPI': ' --direct_prompt_injection --observation_prompt_injection'
    }
    return injection_methods.get(injection_method, '')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Load YAML config file')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    # Load configuration from YAML file
    cfg = load_config(args.cfg_path)

    # Validate required configuration values
    validate_config(cfg)

    # Get API key from user
    api_key = get_api_key()

    # Iterate over attack tool types, LLMs, and attack types
    for attack_tool_type in cfg.get('attack_tool', ['all']):
        for llm_config in cfg['llms']:
            # Handle both string and dict formats for LLM configuration
            if isinstance(llm_config, dict):
                llm_name = llm_config.get('name', '')
                backend = llm_config.get('backend', None)
                base_url = llm_config.get('base_url', None)
            else:
                llm_name = llm_config
                backend = None
                base_url = None

            for attack_type in cfg['attack_types']:
                # Generate model ID, log path, and database path
                model_id = llm_name.split("/")[-1].replace(":", "_").replace(".", "_")
                log_path = f'logs/{cfg.get("injection_method", "direct_prompt_injection")}/{model_id}'
                database = f'memory_db/direct_prompt_injection/{attack_type}_{model_id}'

                # Determine attacker tools path based on attack tool type
                attacker_tools_path = determine_attacker_tools_path(attack_tool_type)

                # Generate log file path and create directory if necessary
                log_memory_type = 'new_memory' if cfg.get('read_db', None) else 'no_memory'
                log_base = f'{log_path}/{cfg.get("defense_type", None)}' if cfg.get('defense_type', None) else f'{log_path}/{log_memory_type}'
                log_file = f'{log_base}/{attack_type}-{attack_tool_type}'
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

                # Construct base command for main attacker script
                base_cmd = construct_base_command(llm_name, api_key, base_url, attack_type, backend, attacker_tools_path, log_file, cfg.get('suffix', ''), database, cfg.get('write_db', None), cfg.get('read_db', None), cfg.get('defense_type', None))

                # Add injection method to command
                specific_cmd = add_injection_method_to_command(cfg.get('injection_method', 'direct_prompt_injection'))

                # Construct final command and execute it
                cmd = f"{base_cmd}{specific_cmd} > {log_file}_{cfg.get('suffix', '')}.log 2>&1 &"
                print(f'{log_file}_{cfg.get("suffix", "")}.log')
                os.system(cmd)

if __name__ == '__main__':
    main()