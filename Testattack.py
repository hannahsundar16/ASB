# This is a just a sample code which load config data points (agent name, dataset path, attacker trigger prompt)
# I'm then loading the stock sythesized dataset which has PII information
# THEN I'M using a stockagent attack agent which is intialized with a tool and task
# the injected trigger string is evaluated/sent to LLM to simulate a data leak attempt
# finally i capture and save whether the attack was successful (eg PII data being leaked) and write results into json file


import argparse
import pandas as pd
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Call LLM with injected string
def call_llm(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"[Error] {e}"

# 🔓 Mock tool interface now replaced with LLM call
class StockAnalyzerTool:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.description = "Analyze the stock dataset"

    def __call__(self, trigger_string):
        print(f"[DEBUG] Sending to LLM: {trigger_string}")
        return call_llm(trigger_string)

# 🧪 Mock agent with tool injection
class StockAgentAttack:
    def __init__(self, agent_name, task_input, args, attacker_tool):
        self.agent_name = agent_name
        self.task_input = task_input
        self.args = args
        self.attacker_tool = attacker_tool
        self.tools = []

    def add_tool(self, tool):
        self.tools.append(tool)

    def run(self):
        messages = []
        messages.append(f"[Injected] Task received: {self.task_input}")
        messages.append(f"[Injected] Trigger: {self.args.trigger}")

        if self.tools:
            result = self.tools[0](self.args.trigger)
            messages.append(f"[Injected] Tool Result: {result}")
        else:
            messages.append("[Error] No tools available.")
        return messages

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, default="gpt-4o")
    parser.add_argument('--shared_jsonl_path', type=str, default="data/stockagent_attack_tools.jsonl")
    parser.add_argument('--trigger', type=str, default=":)")
    parser.add_argument('--res_file', type=str, default="stockagent_results.json")
    return parser.parse_args()

def run_single_attack():
    args = parse_args()
    load_dotenv()

    print(f"[INFO] Reading shared config from: {args.shared_jsonl_path}")
    entries = pd.read_json(args.shared_jsonl_path, lines=True)

    entry = entries.iloc[0]
    agent_name = entry.get("agent_name", "stockagent")
    task = entry.get("tasks", ["Analyze suspicious activity in stock transactions."])[0]

    print(f"\n[\u2713] Running attack for agent: {agent_name}")
    print(f"    \u27a4 Task: {task}")
    print(f"    \u27a4 Entry: {entry.to_dict()}")

    agent = StockAgentAttack(
        agent_name=agent_name,
        task_input=task,
        args=args,
        attacker_tool=entry
    )

    tool = StockAnalyzerTool(csv_path="synthetic_stock_dataset.csv")
    agent.add_tool(tool)

    result = agent.run()

    if os.path.dirname(args.res_file):
        os.makedirs(os.path.dirname(args.res_file), exist_ok=True)
    with open(args.res_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"\n[\u2713] Attack run completed. Output saved to: {args.res_file}")
    print("\n--- Agent Messages ---")
    for msg in result:
        print(msg)

if __name__ == "__main__":
    run_single_attack()

