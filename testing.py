import json
import os
import argparse

def load_task_data(tasks_path):
    """Load task data from a JSONL file."""
    with open(tasks_path, 'r') as f:
        task_data = [json.loads(line) for line in f]
    return task_data

def load_attacker_tools_data(attacker_tools_path):
    """Load attacker tools data from a JSONL file."""
    with open(attacker_tools_path, 'r') as f:
        attacker_tools_data = [json.loads(line) for line in f]
    return attacker_tools_data

def load_vector_db(database_path):
    """Load the Vector DB from a directory."""
    # Replace this with your actual Vector DB loading code
    return {}

def generate_agent_tasks(task_data, attacker_tools_data, vector_db):
    """Generate agent tasks based on the task data, attacker tools data, and Vector DB."""
    agent_tasks = []
    for task in task_data:
        for tool in task['tools']:
            for attacker_tool in attacker_tools_data:
                if attacker_tool['Tool Name'] in tool:
                    agent_task = {
                        'task_name': task['tasks'][0],
                        'tool_name': tool,
                        'attacker_tool_name': attacker_tool['Attacker Tool'],
                        'attack_goal': attacker_tool['Attack goal'],
                        'attacker_instruction': attacker_tool['Attacker Instruction']
                    }
                    agent_tasks.append(agent_task)
    return agent_tasks

def main():
    parser = argparse.ArgumentParser(description='Stock Agent LLaMA')
    parser.add_argument('--res_file', type=str, required=True, help='Path to the results file')
    parser.add_argument('--attacker_tools_path', type=str, required=True, help='Path to the attacker tools data')
    parser.add_argument('--tasks_path', type=str, required=True, help='Path to the task data')
    parser.add_argument('--database', type=str, required=True, help='Path to the Vector DB')
    parser.add_argument('--task_num', type=int, default=10, help='Number of tasks to generate')
    parser.add_argument('--llm_name', type=str, default='meta-llama/Llama-3.3-70B-Instruct', help='Name of the LLaMA model')
    parser.add_argument('--max_gpu_memory', type=int, default=20480, help='Maximum GPU memory to use')
    parser.add_argument('--eval_device', type=str, default='cuda', help='Device to use for evaluation')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate')
    parser.add_argument('--scheduler_log_mode', type=str, default='console', help='Log mode for the scheduler')
    parser.add_argument('--agent_log_mode', type=str, default='console', help='Log mode for the agent')
    parser.add_argument('--llm_kernel_log_mode', type=str, default='console', help='Log mode for the LLaMA kernel')
    parser.add_argument('--pot_backdoor', action='store_true', help='Enable potential backdoor')
    parser.add_argument('--read_db', action='store_true', help='Read from the database')
    parser.add_argument('--tools_info', type=str, help='Path to the tools info file')

    args = parser.parse_args()

    print("Loading task data...")
    task_data = load_task_data(args.tasks_path)
    print("Task data loaded:", task_data)

    print("Loading attacker tools data...")
    attacker_tools_data = load_attacker_tools_data(args.attacker_tools_path)
    print("Attacker tools data loaded:", attacker_tools_data)

    print("Loading Vector DB...")
    vector_db = load_vector_db(args.database)
    print("Vector DB loaded:", vector_db)

    print("Generating agent tasks...")
    agent_tasks = generate_agent_tasks(task_data, attacker_tools_data, vector_db)
    print("Agent tasks generated:", agent_tasks)

    with open(args.res_file, 'w') as f:
        json.dump({'agent_tasks': agent_tasks}, f)

if __name__ == '__main__':
    main()