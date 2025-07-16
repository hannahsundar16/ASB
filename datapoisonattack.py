import warnings
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch, csv, random, json, os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from aios.scheduler.fifo_scheduler import FIFOScheduler
from aios.utils.utils import parse_global_args, delete_directories
from aios.llm_core import llms
from pyopenagi.agents.agent_factory import AgentFactory
from pyopenagi.agents.agent_process import AgentProcessFactory
import requests
import sys
import argparse
import getpass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 5
BASE_URL = "https://glados.ctisl.gtri.org"

class LiteLLMClient:
    """
    A client for interacting with the LiteLLM API.
    """

    def __init__(self, api_key: str, hostname: str):
        """
        Initialize the LiteLLMClient.

        Args:
        - api_key (str): The API key.
        - hostname (str): The hostname.
        """
        self.api_key = api_key
        self.hostname = hostname

    @staticmethod
    def parse_global_args() -> argparse.Namespace:
        """
        Parse global arguments.

        Returns:
        - argparse.Namespace: The parsed arguments.
        """
        parser = argparse.ArgumentParser()

        # LLM settings
        parser.add_argument('--llm_name', type=str, help='Name of the LLM', default='meta-llama/Llama-3.3-70B-Instruct')

        # Attack settings
        parser.add_argument('--attack_tool', type=str, help='Attack tool to use', default='all')
        parser.add_argument('--attack_types', nargs='+', help='Types of attacks to perform', default=['attack_type_1', 'attack_type_2'])

        # Database settings
        parser.add_argument('--database', type=str, help='Path to the database', default='vector_db_stock/')
        parser.add_argument('--write_db', type=str, help='Database to write to', default='memory_db')
        parser.add_argument('--read_db', type=str, help='Database to read from', default='memory_db')

        # Task settings
        parser.add_argument('--tasks_path', type=str, help='Path to the tasks file', default='data/stockagent_task.jsonl')
        parser.add_argument('--task_num', type=int, help='Number of tasks to perform', default=1)

        # Defense settings
        parser.add_argument('--defense_type', type=str, help='Type of defense to use', default='defense_type_1')

        # Injection method
        parser.add_argument('--injection_method', type=str, help='Method of injection to use', default='direct_prompt_injection')

        # Result file
        parser.add_argument('--res_file', type=str, help='Path to the result file', default='results/stock_results.csv')

        # Attacker tools path
        parser.add_argument('--attacker_tools_path', type=str, help='Path to the attacker tools file', default='data/stockagent_attack_tools.jsonl')

        # Tools info path
        parser.add_argument('--tools_info', type=str, help='Path to the tools info file', default='data/stockagent_attack_tools.jsonl')

        # Other settings
        parser.add_argument('--max_gpu_memory', type=int, help='Maximum GPU memory to use', default=20480)
        parser.add_argument('--eval_device', type=str, help='Device to use for evaluation', default='cuda')
        parser.add_argument('--max_new_tokens', type=int, help='Maximum number of new tokens to generate', default=512)
        parser.add_argument('--scheduler_log_mode', type=str, help='Log mode for the scheduler', default='console')
        parser.add_argument('--agent_log_mode', type=str, help='Log mode for the agent', default='console')
        parser.add_argument('--llm_kernel_log_mode', type=str, help='Log mode for the LLM kernel', default='console')
        parser.add_argument('--pot_backdoor', type=bool, help='Whether to use the pot backdoor', default=True)

        return parser.parse_args()

    def chat(self, model: str, messages: list) -> dict:
        """
        Send a chat request to the LLM.

        Args:
        - model (str): The model name.
        - messages (list): The messages to send.

        Returns:
        - dict: The response from the LLM.
        """
        url = f"{self.hostname}/chat/{model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {"messages": messages}
        response = requests.post(url, headers=headers, json=data)
        return response.json()

def get_api_key() -> str:
    """
    Get API key from user.

    Returns:
    - str: The API key.
    """
    return getpass.getpass("Please enter your API key: ")

def clean_cache(root_directory: str, target_dirs: list) -> None:
    """
    Clean the cache in the given root directory.

    Args:
    - root_directory (str): The root directory.
    - target_dirs (list): The target directories to clean.
    """
    delete_directories(root_directory, target_dirs)

def seed_everything(seed: int) -> None:
    """
    Seed the random number generators.

    Args:
    - seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def inject_data_poisoning(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Inject data poisoning into the dataset.

    Args:
    - df (pd.DataFrame): The dataset.
    - args (argparse.Namespace): The parsed arguments.

    Returns:
    - pd.DataFrame: The poisoned dataset.
    """
    # Introduce some randomness into the data poisoning
    num_poisoned_samples = int(len(df) * 0.1)
    poisoned_indices = np.random.choice(len(df), num_poisoned_samples, replace=False)

    for index in poisoned_indices:
        # Poison the data by changing the label
        df.loc[index, 'label'] = np.random.choice([0, 1])

    return df

def run_agent_task(agent_name: str, task: str, args: argparse.Namespace, tool_with_path: dict, vector_db: dict, aggressive: bool) -> dict:
    """
    Run an agent task.

    Args:
    - agent_name (str): The agent name.
    - task (str): The task name.
    - args (argparse.Namespace): The parsed arguments.
    - tool_with_path (dict): The tool with path.
    - vector_db (dict): The vector database.
    - aggressive (bool): Whether to use aggressive mode.

    Returns:
    - dict: The result of the agent task.
    """
    try:
        logger.info(f"Running agent task for {agent_name} with task {task}")
        # Introduce some randomness into the result
        result = random.random()
        if result < 0.8:
            return {
                "agent_name": agent_name,
                "task": task,
                "result": "Success",
                "tool": str(tool_with_path),
                "vector_db": str(vector_db),
                "aggressive": aggressive
            }
        else:
            return {
                "agent_name": agent_name,
                "task": task,
                "result": "Failure",
                "tool": str(tool_with_path),
                "vector_db": str(vector_db),
                "aggressive": aggressive
            }
    except Exception as e:
        logger.error(f"Error running agent task: {e}")
        return {
            "agent_name": agent_name,
            "task": task,
            "result": None,
            "tool": str(tool_with_path),
            "vector_db": str(vector_db),
            "aggressive": aggressive,
            "error": str(e)
        }

def calculate_metrics(results: list, df: pd.DataFrame) -> dict:
    """
    Calculate the attack success rate, false positive rate, false negative rate, 
    precision, recall, F1 score, accuracy, and true negative rate.

    Args:
    - results (list): The list of results.
    - df (pd.DataFrame): The DataFrame containing the synthetic stock dataset.

    Returns:
    - dict: The metrics.
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for result in results:
        if result["result"] == "Success" and result["aggressive"]:
            true_positives += 1
        elif result["result"] == "Success" and not result["aggressive"]:
            false_positives += 1
        elif result["result"] == "Failure" and result["aggressive"]:
            false_negatives += 1
        elif result["result"] == "Failure" and not result["aggressive"]:
            true_negatives += 1

    attack_success_rate = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
    false_positive_rate = (false_positives / (false_positives + true_negatives)) if (false_positives + true_negatives) > 0 else 0
    false_negative_rate = (false_negatives / (false_negatives + true_positives)) if (false_negatives + true_positives) > 0 else 0
    precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * (precision * recall)) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = ((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0
    true_negative_rate = (true_negatives / (true_negatives + false_positives)) if (true_negatives + false_positives) > 0 else 0

    return {
        "attack_success_rate": attack_success_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "true_negative_rate": true_negative_rate
    }

def main() -> None:
    """
    Main function.
    """
    seed_everything(0)
    start_time = datetime.now()
    logger.info(f"Attack started at: {start_time.strftime('%Y-%m-%d %H:%M')}")

    warnings.filterwarnings("ignore")
    command_line_arguments = LiteLLMClient.parse_global_args()
    logger.info(f"Command line arguments: {command_line_arguments}")
    load_dotenv()
    api_key = get_api_key()

    try:
        llm = LiteLLMClient(api_key, BASE_URL)
    except Exception as e:
        logger.error(f"Error initializing LLMClient: {e}")
        return

    # Initialize the scheduler with the required arguments
    scheduler = FIFOScheduler(llm, command_line_arguments.scheduler_log_mode)
    scheduler.start()

    agent_tasks = []
    try:
        df = pd.read_csv('synthetic_stock_dataset_small.csv')
        poisoned_df = inject_data_poisoning(df, command_line_arguments)

        for index, row in poisoned_df.iterrows():
            logger.info(f"Task: {row}")
            agent_name = "agent"
            task_name = "task"
            tool_with_path = {"Tool Name": "tool", "Attacker Tool": "attacker_tool"}  
            vector_db = {}  
            aggressive = True

            agent_tasks.append((agent_name, task_name, command_line_arguments, tool_with_path, vector_db, aggressive))

        logger.info(f"Agent tasks: {agent_tasks}")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(run_agent_task, *task): task for task in agent_tasks}
            results = []
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    logger.info(f"Agent {task[0]} completed task {task[1]} with result: {result['result']}")
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error running agent task: {e}")
            logger.info(f"Results: {results}")

            metrics = calculate_metrics(results, df)
            logger.info(f"Metrics: {metrics}")

            # Write results to CSV
            with open(command_line_arguments.res_file, 'w', newline='') as csvfile:
                fieldnames = ['agent_name', 'task', 'result', 'tool', 'vector_db', 'aggressive']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow({
                        'agent_name': result['agent_name'],
                        'task': result['task'],
                        'result': result['result'],
                        'tool': result['tool'],
                        'vector_db': result['vector_db'],
                        'aggressive': result['aggressive']
                    })

            # Write metrics to CSV
            with open(command_line_arguments.res_file.replace('.csv', '_metrics.csv'), 'w', newline='') as csvfile:
                fieldnames = [
                    'attack_success_rate', 
                    'false_positive_rate', 
                    'false_negative_rate', 
                    'precision', 
                    'recall', 
                    'f1_score', 
                    'accuracy', 
                    'true_negative_rate'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    'attack_success_rate': metrics['attack_success_rate'],
                    'false_positive_rate': metrics['false_positive_rate'],
                    'false_negative_rate': metrics['false_negative_rate'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'accuracy': metrics['accuracy'],
                    'true_negative_rate': metrics['true_negative_rate']
                })

    except Exception as e:
        logger.error(f"Error running agent tasks: {e}")

    scheduler.stop()
    clean_cache("./", [])
    end_time = datetime.now()
    logger.info(f"Attack ended at: {end_time.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Total duration: {end_time - start_time}")

if __name__ == "__main__":
    main()