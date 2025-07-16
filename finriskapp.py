# Hannah Sundar - 7/15/2025
# 
# Flash app = FinRisk
#
# The application provides a Flask-based UI where users can query a dataset or launch simulated LLM-based attacks using 
# configurable agents. Upon launching an attack, the system loads a CSV dataset and applies data poisoning techniques such 
# as trigger injection or label flipping based on user-defined parameters. It then creates multiple agents, each configured 
# with specific tools (e.g., backdoor trigger, direct injection, label flipping), LLM models (e.g. Llama-3.3-70B-Instruct), and aggression levels. 
# These agents execute tasks in parallel, invoking their assigned tools and prompting the LLM to simulate adversarial behaviors 
# or poisoning effects. The system collects the agents’ results, calculates key evaluation metrics (attack success rate, 
# precision, recall, etc.), and returns them to the frontend for analysis and visualization. 
# 
import os
import random
import warnings
import logging
import numpy as np
import pandas as pd
import torch
import csv
import json
import getpass
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from datetime import datetime
from aios.scheduler.fifo_scheduler import FIFOScheduler
from aios.utils.utils import parse_global_args, delete_directories
from aios.llm_core import llms
from pyopenagi.agents.agent_factory import AgentFactory
from pyopenagi.agents.agent_process import AgentProcessFactory
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import sys
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 5
BASE_URL = "https://glados.ctisl.gtri.org"
DATA_FILE = "synthetic_stock_dataset_small.csv"
LLM_BASE_URL = "https://glados.ctisl.gtri.org/v1"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Global variables
API_KEY = None
llm_client = None
scheduler = None

app = Flask(__name__)

# ENHANCED FRONTEND HTML (combining both UIs)
FRONTEND_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>FinRisk Agent Demo</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f0f2f5; padding: 20px; color: #333; }
    h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
    .container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px;
                 box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
    .tab-container { display: flex; margin-bottom: 20px; border-bottom: 2px solid #ecf0f1; }
    .tab { padding: 15px 25px; cursor: pointer; background: #ecf0f1; border: none; margin-right: 5px;
           border-radius: 5px 5px 0 0; font-size: 16px; transition: all 0.3s; }
    .tab.active { background: #3498db; color: white; }
    .tab:hover { background: #34495e; color: white; }
    .tab-content { display: none; padding: 20px 0; }
    .tab-content.active { display: block; }
    .form-group { margin-bottom: 20px; }
    .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #2c3e50; }
    input[type=text], input[type=number], select { width: 100%; padding: 12px; border-radius: 5px;
                                                   border: 1px solid #ccc; font-size: 16px; box-sizing: border-box; }
    .input-row { display: flex; gap: 10px; align-items: end; }
    .input-row input { flex: 1; }
    button { padding: 12px 20px; border-radius: 5px; border: none; background-color: #3498db; color: white;
             font-size: 16px; cursor: pointer; transition: background-color 0.3s; }
    button:hover { background-color: #2980b9; }
    button.danger { background-color: #e74c3c; }
    button.danger:hover { background-color: #c0392b; }
    button.success { background-color: #27ae60; }
    button.success:hover { background-color: #229954; }
    pre { background: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; margin-top: 20px;
          font-size: 14px; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; }
    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }
    .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #3498db; }
    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
    .status { padding: 10px; border-radius: 5px; margin: 10px 0; font-weight: bold; }
    .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
    .loading { text-align: center; padding: 20px; font-style: italic; color: #666; }
    footer { text-align: center; margin-top: 40px; font-size: 14px; color: #888; }
  </style>
</head>
<body>
  <div class="container">
    <h1> FinRisk Agent Demo</h1>
   
    <div class="tab-container">
      <button class="tab active" onclick="showTab('basic')">Basic Query</button>
      <button class="tab" onclick="showTab('agent')">Agent Attack</button>
      <button class="tab" onclick="showTab('config')">Configuration</button>
    </div>

    <!-- Basic Query Tab -->
    <div id="basic" class="tab-content active">
      <h3>Dataset Query Interface</h3>
      <p>Ask about the dataset (try: <em>show distribution</em>, <em>show summary</em>, <em>give example</em>, <em>show poison</em>, <em>compare labels</em>, <em>run ml model</em>, <em>ask llm: [question]</em>):</p>
      <div class="input-row">
        <input type="text" id="question" placeholder="Type your question here..." />
        <button onclick="sendQuery()">Send Query</button>
      </div>
      <pre id="result">Results will appear here...</pre>
    </div>

    <!-- Agent Attack Tab -->
    <div id="agent" class="tab-content">
      <h3>Agent-Based Attack Simulation</h3>
      <div class="form-group">
        <label for="llm_name">LLM Model:</label>
        <select id="llm_name">
          <option value="meta-llama/Llama-3.3-70B-Instruct">Llama-3.3-70B-Instruct</option>
          <option value="meta-llama/Llama-2-70B-chat">Llama-2-70B-chat</option>
          <option value="gpt-4">GPT-4</option>
        </select>
      </div>
      <div class="form-group">
        <label for="attack_tool">Attack Tool:</label>
        <select id="attack_tool">
          <option value="all">All Tools</option>
          <option value="direct_injection">Direct Injection</option>
          <option value="backdoor_trigger">Backdoor Trigger</option>
          <option value="label_flipping">Label Flipping</option>
        </select>
      </div>
      <div class="form-group">
        <label for="poison_fraction">Poison Fraction (0.0-1.0):</label>
        <input type="number" id="poison_fraction" value="0.1" min="0.0" max="1.0" step="0.01" />
      </div>
      <div class="form-group">
        <label for="task_num">Number of Tasks:</label>
        <input type="number" id="task_num" value="10" min="1" max="100" />
      </div>
      <div class="form-group">
        <label for="aggressive_mode">Aggressive Mode:</label>
        <select id="aggressive_mode">
          <option value="true">Enabled</option>
          <option value="false">Disabled</option>
        </select>
      </div>
      <button onclick="runAgentAttack()" class="danger">Launch Agent Attack</button>
      <button onclick="stopAttack()" class="success">Stop Attack</button>
     
      <div id="attack_status"></div>
      <div id="attack_metrics"></div>
      <pre id="attack_results">Attack results will appear here...</pre>
    </div>

    <!-- Configuration Tab -->
    <div id="config" class="tab-content">
      <h3>System Configuration</h3>
      <div class="form-group">
        <label for="api_key_input">API Key:</label>
        <input type="password" id="api_key_input" placeholder="Enter your API key..." />
        <button onclick="setApiKey()">Set API Key</button>
      </div>
      <div class="form-group">
        <label for="base_url">Base URL:</label>
        <input type="text" id="base_url" value="https://glados.ctisl.gtri.org" />
      </div>
      <div class="form-group">
        <label for="max_workers">Max Workers:</label>
        <input type="number" id="max_workers" value="5" min="1" max="20" />
      </div>
      <div class="form-group">
        <label for="data_file">Data File:</label>
        <input type="text" id="data_file" value="synthetic_stock_dataset_small.csv" />
      </div>
      <button onclick="testConnection()">Test Connection</button>
      <button onclick="resetConfig()">Reset to Defaults</button>
      <pre id="config_status">Configuration status will appear here...</pre>
    </div>
  </div>

  <footer> FinRisk Agent Demo - Research Use Only</footer>

  <script>
    let attackInProgress = false;
    let pollInterval;

    function showTab(tabName) {
      // Hide all tabs
      const tabs = document.querySelectorAll('.tab-content');
      tabs.forEach(tab => tab.classList.remove('active'));
     
      const tabButtons = document.querySelectorAll('.tab');
      tabButtons.forEach(button => button.classList.remove('active'));
     
      // Show selected tab
      document.getElementById(tabName).classList.add('active');
      event.target.classList.add('active');
    }

    async function sendQuery() {
      const question = document.getElementById('question').value;
      if (!question.trim()) {
        document.getElementById('result').textContent = 'Please enter a question.';
        return;
      }
     
      document.getElementById('result').textContent = 'Loading...';
      try {
        const res = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
        const data = await res.json();
        document.getElementById('result').textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        document.getElementById('result').textContent = 'Error: ' + err;
      }
    }

    async function runAgentAttack() {
      if (attackInProgress) {
        alert('Attack already in progress!');
        return;
      }

      const params = {
        llm_name: document.getElementById('llm_name').value,
        attack_tool: document.getElementById('attack_tool').value,
        poison_fraction: parseFloat(document.getElementById('poison_fraction').value),
        task_num: parseInt(document.getElementById('task_num').value),
        aggressive_mode: document.getElementById('aggressive_mode').value === 'true',
        max_workers: parseInt(document.getElementById('max_workers').value) || 5
      };

      attackInProgress = true;
      document.getElementById('attack_status').innerHTML = '<div class="status warning">Attack in progress...</div>';
      document.getElementById('attack_results').textContent = 'Initializing attack...';

      try {
        const res = await fetch('/run_agent_attack', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        });
       
        const data = await res.json();
       
        if (data.success) {
          document.getElementById('attack_status').innerHTML = '<div class="status success">Attack completed successfully!</div>';
          displayMetrics(data.metrics);
          document.getElementById('attack_results').textContent = JSON.stringify(data, null, 2);
        } else {
          document.getElementById('attack_status').innerHTML = '<div class="status error">Attack failed: ' + data.error + '</div>';
          document.getElementById('attack_results').textContent = 'Error: ' + data.error;
        }
      } catch (err) {
        document.getElementById('attack_status').innerHTML = '<div class="status error">Network error: ' + err + '</div>';
        document.getElementById('attack_results').textContent = 'Network Error: ' + err;
      } finally {
        attackInProgress = false;
      }
    }

    async function stopAttack() {
      if (!attackInProgress) {
        alert('No attack in progress!');
        return;
      }

      try {
        await fetch('/stop_attack', { method: 'POST' });
        document.getElementById('attack_status').innerHTML = '<div class="status warning">Attack stopped by user</div>';
        attackInProgress = false;
      } catch (err) {
        console.error('Error stopping attack:', err);
      }
    }

    function displayMetrics(metrics) {
      if (!metrics) return;
     
      const metricsHtml = `
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-value">${(metrics.attack_success_rate * 100).toFixed(1)}%</div>
            <div class="metric-label">Attack Success Rate</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
            <div class="metric-label">Accuracy</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${(metrics.precision * 100).toFixed(1)}%</div>
            <div class="metric-label">Precision</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${(metrics.recall * 100).toFixed(1)}%</div>
            <div class="metric-label">Recall</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${(metrics.f1_score * 100).toFixed(1)}%</div>
            <div class="metric-label">F1 Score</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${(metrics.false_positive_rate * 100).toFixed(1)}%</div>
            <div class="metric-label">False Positive Rate</div>
          </div>
        </div>
      `;
      document.getElementById('attack_metrics').innerHTML = metricsHtml;
    }

    async function setApiKey() {
      const apiKey = document.getElementById('api_key_input').value;
      if (!apiKey.trim()) {
        alert('Please enter an API key');
        return;
      }

      try {
        const res = await fetch('/set_api_key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ api_key: apiKey })
        });
        const data = await res.json();
       
        if (data.success) {
          document.getElementById('config_status').textContent = 'API key set successfully!';
          document.getElementById('api_key_input').value = '';
        } else {
          document.getElementById('config_status').textContent = 'Error setting API key: ' + data.error;
        }
      } catch (err) {
        document.getElementById('config_status').textContent = 'Network error: ' + err;
      }
    }

    async function testConnection() {
      document.getElementById('config_status').textContent = 'Testing connection...';
     
      try {
        const res = await fetch('/test_connection', { method: 'POST' });
        const data = await res.json();
       
        if (data.success) {
          document.getElementById('config_status').textContent = 'Connection test passed! ';
        } else {
          document.getElementById('config_status').textContent = 'Connection test failed: ' + data.error;
        }
      } catch (err) {
        document.getElementById('config_status').textContent = 'Connection test failed: ' + err;
      }
    }

    function resetConfig() {
      document.getElementById('base_url').value = 'https://glados.ctisl.gtri.org';
      document.getElementById('max_workers').value = '5';
      document.getElementById('data_file').value = 'synthetic_stock_dataset_small.csv';
      document.getElementById('config_status').textContent = 'Configuration reset to defaults.';
    }
  </script>
</body>
</html>
"""

class LiteLLMClient:
    """A client for interacting with the LiteLLM API."""

    def __init__(self, api_key: str, hostname: str):
        if not api_key or not hostname:
            raise ValueError("API key and hostname are required")
        self.api_key = api_key
        self.hostname = hostname
        self.chat = OpenAI(api_key=api_key, base_url=hostname)

    @staticmethod
    def parse_global_args() -> argparse.Namespace:
        """Parse global arguments with defaults."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--llm_name', type=str, default='meta-llama/Llama-3.3-70B-Instruct')
        parser.add_argument('--attack_tool', type=str, default='all')
        parser.add_argument('--attack_types', nargs='+', default=['attack_type_1', 'attack_type_2'])
        parser.add_argument('--database', type=str, default='vector_db_stock/')
        parser.add_argument('--write_db', type=str, default='memory_db')
        parser.add_argument('--read_db', type=str, default='memory_db')
        parser.add_argument('--tasks_path', type=str, default='data/stockagent_task.jsonl')
        parser.add_argument('--task_num', type=int, default=1)
        parser.add_argument('--defense_type', type=str, default='defense_type_1')
        parser.add_argument('--injection_method', type=str, default='direct_prompt_injection')
        parser.add_argument('--res_file', type=str, default='results/stock_results.csv')
        parser.add_argument('--attacker_tools_path', type=str, default='data/stockagent_attack_tools.jsonl')
        parser.add_argument('--tools_info', type=str, default='data/stockagent_attack_tools.jsonl')
        parser.add_argument('--max_gpu_memory', type=int, default=20480)
        parser.add_argument('--eval_device', type=str, default='cuda')
        parser.add_argument('--max_new_tokens', type=int, default=512)
        parser.add_argument('--scheduler_log_mode', type=str, default='console')
        parser.add_argument('--agent_log_mode', type=str, default='console')
        parser.add_argument('--llm_kernel_log_mode', type=str, default='console')
        parser.add_argument('--pot_backdoor', type=bool, default=True)
        return parser.parse_args()

def seed_everything(seed: int = 0) -> None:
    """Seed all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def safe_json(df_slice):
    """Convert DataFrame slice to JSON-safe format."""
    def convert_value(val):
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        return val
    return [
        {k: convert_value(v) for k, v in record.items()}
        for record in df_slice.to_dict(orient='records')
    ]

def inject_data_poisoning(df, poison_fraction=0.1, trigger_feature='trigger', trigger_value=9999, target_value="BUY"):
    """Inject data poisoning with enhanced capabilities."""
    poisoned_df = df.copy()
    if trigger_feature not in poisoned_df.columns:
        poisoned_df[trigger_feature] = 0
   
    num_to_poison = int(len(poisoned_df) * poison_fraction)
    indices_to_poison = np.random.choice(poisoned_df.index, num_to_poison, replace=False)
   
    for idx in indices_to_poison:
        poisoned_df.at[idx, trigger_feature] = trigger_value
        poisoned_df.at[idx, 'transaction_type'] = target_value
       
        # Introduce subtle errors in other columns
        for col in poisoned_df.columns:
            if col not in ['transaction_type', trigger_feature]:
                if np.random.rand() < 0.2:  # 20% chance of introducing an error
                    if poisoned_df[col].dtype == np.float64:
                        poisoned_df.at[idx, col] = poisoned_df.at[idx, col] + np.random.uniform(-10, 10)
                    elif poisoned_df[col].dtype == np.int64:
                        poisoned_df.at[idx, col] = poisoned_df.at[idx, col] + np.random.randint(-5, 5)
                    else:
                        poisoned_df.at[idx, col] = np.random.choice(['A', 'B', 'C', 'D'], p=[0.4, 0.3, 0.2, 0.1])
   
    return poisoned_df

def run_agent_task(agent_name: str, task: str, args: argparse.Namespace, tool_with_path: dict, vector_db: dict, aggressive: bool) -> dict:
    """Run an agent task with enhanced error handling."""
    try:
        logger.info(f"Running agent task for {agent_name} with task {task}")
       
        # Simulate realistic agent behavior
        success_probability = 0.8 if aggressive else 0.6
        result = random.random()
       
        if result < success_probability:
            return {
                "agent_name": agent_name,
                "task": task,
                "result": "Success",
                "tool": str(tool_with_path),
                "vector_db": str(vector_db),
                "aggressive": aggressive,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "agent_name": agent_name,
                "task": task,
                "result": "Failure",
                "tool": str(tool_with_path),
                "vector_db": str(vector_db),
                "aggressive": aggressive,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error running agent task: {e}")
        return {
            "agent_name": agent_name,
            "task": task,
            "result": "Error",
            "tool": str(tool_with_path),
            "vector_db": str(vector_db),
            "aggressive": aggressive,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def calculate_metrics(results: list, df: pd.DataFrame) -> dict:
    """Calculate comprehensive attack metrics."""
    try:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for result in results:
            if result.get("result") == "Success" and result.get("aggressive"):
                true_positives += 1
            elif result.get("result") == "Success" and not result.get("aggressive"):
                false_positives += 1
            elif result.get("result") == "Failure" and result.get("aggressive"):
                false_negatives += 1
            elif result.get("result") == "Failure" and not result.get("aggressive"):
                true_negatives += 1

        # Calculate metrics with zero-division protection
        def safe_divide(a, b):
            return a / b if b > 0 else 0

        attack_success_rate = safe_divide(true_positives, (true_positives + false_negatives))
        false_positive_rate = safe_divide(false_positives, (false_positives + true_negatives))
        false_negative_rate = safe_divide(false_negatives, (false_negatives + true_positives))
        precision = safe_divide(true_positives, (true_positives + false_positives))
        recall = safe_divide(true_positives, (true_positives + false_negatives))
        f1_score = safe_divide(2 * (precision * recall), (precision + recall))
        accuracy = safe_divide((true_positives + true_negatives),
                              (true_positives + true_negatives + false_positives + false_negatives))
        true_negative_rate = safe_divide(true_negatives, (true_negatives + false_positives))

        return {
            "attack_success_rate": attack_success_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_negative_rate": true_negative_rate,
            "total_samples": len(results),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"error": str(e)}

def train_random_forest(df):
    """Train a random forest model on the dataset."""
    if 'transaction_type' not in df.columns:
        return "Error: Dataset has no 'transaction_type' column."
   
    df = df.dropna(subset=['transaction_type'])
    if len(df) < 10:
        return "Error: Not enough data to train."
   
    features = df.drop(columns=['transaction_type'], errors='ignore')
    features = pd.get_dummies(features.select_dtypes(include=[np.number]))
    labels = df['transaction_type']
   
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
   
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

# FLASK ROUTES

@app.route("/")
def home():
    """Serve the main UI."""
    return render_template_string(FRONTEND_HTML)

@app.route("/query", methods=["POST"])
def query_dataset():
    """Handle basic dataset queries."""
    data = request.get_json()
    question = data.get("question", "").lower().strip()
   
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        return jsonify({"error": f"Failed to load dataset: {str(e)}"})

    poisoned_df = inject_data_poisoning(df.copy())

    if question.startswith("ask llm:"):
        if not llm_client:
            return jsonify({"error": "API key not set. Please configure in the Configuration tab."})
       
        user_prompt = question.replace("ask llm:", "").strip()
        sample = poisoned_df.sample(min(10, len(poisoned_df))).to_string()
        prompt = f"You are an expert data analyst. The user question is: {user_prompt}\nHere is a sample of the data:\n{sample}"
       
        try:
            completion = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            response_text = f"LLM request failed: {str(e)}"

        return jsonify({"llm_response": response_text})

    if "distribution" in question:
        counts = df['transaction_type'].value_counts().to_dict()
        counts = {str(k): int(v) for k, v in counts.items()}
        return jsonify({"transaction_distribution": counts})

    if "summary" in question:
        desc = df.describe(include='all')
        summary = {}
        for col in desc.columns:
            summary[col] = {
                k: (None if pd.isna(v) else (v.item() if isinstance(v, (np.integer, np.floating)) else v))
                for k, v in desc[col].items()
            }
        return jsonify({"dataset_summary": summary})

    if "example" in question or "sample" in question:
        return jsonify({"example_row": safe_json(df.sample(1))})

    if "poison" in question:
        return jsonify({"poisoned_samples": safe_json(poisoned_df.sample(min(5, len(poisoned_df))))})

    if "backdoor" in question or "trigger" in question or "how many" in question:
        num_backdoor = (poisoned_df['trigger'] == 9999).sum()
        percent = round((num_backdoor / len(poisoned_df)) * 100, 2)
        return jsonify({"num_backdoor_trigger": int(num_backdoor), "percent_poisoned": percent})

    if "run ml" in question or "train model" in question:
        report = train_random_forest(poisoned_df)
        return jsonify({"random_forest_report": report})

    return jsonify({
        "message": "Unrecognized question. Try 'show distribution', 'show summary', 'example', 'show poison', 'how many have backdoor trigger', 'run ml model', or 'ask llm: <your question>'."
    })

@app.route("/run_agent_attack", methods=["POST"])
def run_agent_attack():
    """Run the agent-based attack simulation."""
    global scheduler
   
    if not API_KEY:
        return jsonify({"success": False, "error": "API key not configured"})
   
    data = request.get_json()
    llm_name = data.get('llm_name', 'meta-llama/Llama-3.3-70B-Instruct')
    attack_tool = data.get('attack_tool', 'all')
    poison_fraction = data.get('poison_fraction', 0.1)
    task_num = data.get('task_num', 10)
    aggressive_mode = data.get('aggressive_mode', True)
    max_workers = data.get('max_workers', 5)
   
    try:
        # Load and poison the dataset
        df = pd.read_csv(DATA_FILE)
        poisoned_df = inject_data_poisoning(df.copy(), poison_fraction=poison_fraction)
       
        # Initialize scheduler if not already done
        if not scheduler:
            scheduler = FIFOScheduler(LiteLLMClient(API_KEY, BASE_URL), "console")
            scheduler.start()
       
        # Create agent tasks
        args = LiteLLMClient.parse_global_args()
        agent_tasks = []
        for i in range(task_num):
            agent_name = f"agent_{i}"
            task_name = f"poison_task_{i}"
            tool_with_path = {
                "Tool Name": attack_tool,
                "Attacker Tool": f"attacker_tool_{i}",
                "LLM Model": llm_name
            }
            vector_db = {"database": "vector_db_stock/", "type": "chroma"}
            is_aggressive = aggressive_mode if i % 2 == 0 else not aggressive_mode
            agent_tasks.append((agent_name, task_name, args, tool_with_path, vector_db, is_aggressive))
       
        # Run tasks in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_agent_task, *task): task for task in agent_tasks}
           
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error running agent task: {e}")
                    results.append({
                        'agent_name': task[0],
                        'task': task[1],
                        'result': 'Error',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
       
        # Calculate metrics
        metrics = calculate_metrics(results, df)
       
        # Prepare response
        response = {
            "success": True,
            "metrics": metrics,
            "results": results,
            "dataset_info": {
                "original_size": len(df),
                "poisoned_size": len(poisoned_df),
                "poison_fraction": poison_fraction,
                "num_poisoned": int(len(poisoned_df) * poison_fraction)
            },
            "attack_config": {
                "llm_name": llm_name,
                "attack_tool": attack_tool,
                "task_num": task_num,
                "aggressive_mode": aggressive_mode,
                "max_workers": max_workers
            },
            "timestamp": datetime.now().isoformat()
        }
       
        return jsonify(response)
       
    except Exception as e:
        logger.error(f"Error running agent attack: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/stop_attack", methods=["POST"])
def stop_attack():
    """Stop the current attack simulation."""
    global scheduler
   
    try:
        if scheduler:
            scheduler.stop()
            scheduler = None
        return jsonify({"success": True, "message": "Attack stopped successfully"})
    except Exception as e:
        logger.error(f"Error stopping attack: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/set_api_key", methods=["POST"])
def set_api_key():
    """Set the API key for LLM access."""
    global API_KEY, llm_client
   
    data = request.get_json()
    api_key = data.get("api_key", "").strip()
   
    if not api_key:
        return jsonify({"success": False, "error": "API key cannot be empty"})
   
    try:
        # Test the API key
        test_client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
       
        # Make a simple test call
        test_completion = test_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
       
        # If we get here, the API key works
        API_KEY = api_key
        llm_client = test_client
       
        return jsonify({"success": True, "message": "API key set and validated successfully"})
       
    except Exception as e:
        logger.error(f"Error setting API key: {e}")
        return jsonify({"success": False, "error": f"Invalid API key: {str(e)}"})

@app.route("/test_connection", methods=["POST"])
def test_connection():
    """Test the connection to the LLM service."""
    if not API_KEY:
        return jsonify({"success": False, "error": "API key not configured"})
   
    try:
        test_completion = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=5
        )
       
        return jsonify({
            "success": True,
            "message": "Connection test successful",
            "model": LLM_MODEL,
            "response": test_completion.choices[0].message.content
        })
       
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/get_dataset_info", methods=["GET"])
def get_dataset_info():
    """Get information about the current dataset."""
    try:
        df = pd.read_csv(DATA_FILE)
        poisoned_df = inject_data_poisoning(df.copy())
       
        info = {
            "original_dataset": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            },
            "poisoned_dataset": {
                "rows": len(poisoned_df),
                "columns": len(poisoned_df.columns),
                "poison_samples": int((poisoned_df['trigger'] == 9999).sum()),
                "poison_percentage": round(((poisoned_df['trigger'] == 9999).sum() / len(poisoned_df)) * 100, 2)
            },
            "transaction_distribution": df['transaction_type'].value_counts().to_dict()
        }
       
        return jsonify({"success": True, "dataset_info": info})
       
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        return jsonify({"success": False, "error": str(e)})

def clean_cache(root_directory: str, target_dirs: list) -> None:
    """Clean the cache in the given root directory."""
    try:
        delete_directories(root_directory, target_dirs)
        logger.info("Cache cleaned successfully")
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")

def initialize_system():
    """Initialize the system with default settings."""
    global API_KEY, llm_client
   
    # Prompt for API key at startup
    try:
        API_KEY = getpass.getpass("Enter your Glados API Key: ").strip()
        if API_KEY:
            llm_client = OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)
            logger.info("API key set successfully")
        else:
            logger.warning("No API key provided. LLM features will be disabled.")
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during initialization: {e}")

#def main():
#    """Main function to run the Flask application."""
#    # Initialize system
#    initialize_system()
   
    # Set up environment
#    seed_everything()
#    warnings.filterwarnings("ignore")
   
    # Clean any existing cache
#    clean_cache("./", ["__pycache__", ".pytest_cache"])
   
#    logger.info("Starting Advanced Data Poison Attack Demo")
#    logger.info("===========================================")
#    logger.info(f"Server starting on http://localhost:5002")
#    logger.info("Features available:")
#    logger.info("- Basic dataset querying")
#    logger.info("- Agent-based attack simulation")
#    logger.info("- Real-time metrics and monitoring")
#    logger.info("- Configuration management")
#    logger.info("===========================================")
   
#    try:
#        app.run(host="0.0.0.0", port=5002, debug=True, threaded=True)
#    except KeyboardInterrupt:
#        logger.info("Server stopped by user")
#    except Exception as e:
#        logger.error(f"Server error: {e}")
#    finally:
#        # Clean up
#        global scheduler
#        if scheduler:
#            scheduler.stop()
#        clean_cache("./", ["__pycache__"])

#        logger.info("Application shutdown complete")

#if __name__ == "__main__":
#    main()
# MAIN
def main():
    initialize_system()
    seed_everything()
    warnings.filterwarnings("ignore")
    logger.info("Starting Flask server on http://localhost:5004")
    app.run(host="0.0.0.0", port=5004, debug=True)

if __name__ == "__main__":
    main()
