import subprocess
import pandas as pd
import re

def get_gpu_memory_usage():
    """
    Get detailed GPU memory usage information using nvidia-smi.
    Returns a DataFrame with process information and memory usage.
    """
    try:
        # Run nvidia-smi command to get GPU processes and their memory usage
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory,name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        
        # Parse the output
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) >= 3:
                    pid = int(parts[0])
                    # Extract memory value and convert to MiB
                    memory_str = parts[1]
                    memory_value = int(re.search(r'(\d+)', memory_str).group(1))
                    process_name = parts[2]
                    
                    # Get more details about the process
                    try:
                        proc_info = subprocess.run(
                            ['ps', '-p', str(pid), '-o', 'user,cmd', '--no-headers'],
                            capture_output=True, text=True, check=True
                        )
                        proc_details = proc_info.stdout.strip().split(maxsplit=1)
                        user = proc_details[0] if proc_details else "Unknown"
                        cmd = proc_details[1] if len(proc_details) > 1 else "Unknown"
                    except:
                        user = "Unknown"
                        cmd = "Unknown"
                    
                    processes.append({
                        'PID': pid,
                        'User': user,
                        'Process Name': process_name,
                        'Memory Usage (MiB)': memory_value,
                        'Command': cmd
                    })
        
        # Get total GPU memory information
        total_info = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        
        total_parts = total_info.stdout.strip().split(', ')
        total_memory = int(re.search(r'(\d+)', total_parts[0]).group(1))
        used_memory = int(re.search(r'(\d+)', total_parts[1]).group(1))
        free_memory = int(re.search(r'(\d+)', total_parts[2]).group(1))
        
        # Create DataFrame
        df = pd.DataFrame(processes)
        
        return df, total_memory, used_memory, free_memory
    
    except subprocess.CalledProcessError:
        print("Error: nvidia-smi command failed. Make sure NVIDIA drivers are installed properly.")
        return pd.DataFrame(), 0, 0, 0
    except FileNotFoundError:
        print("Error: nvidia-smi command not found. Make sure NVIDIA drivers are installed.")
        return pd.DataFrame(), 0, 0, 0

def main():
    print("Analyzing GPU Memory Usage...\n")
    
    # Get GPU memory usage
    df, total_memory, used_memory, free_memory = get_gpu_memory_usage()
    
    if df.empty:
        print("No GPU processes found or unable to get GPU information.")
        return
    
    # Print summary
    print(f"Total GPU Memory: {total_memory} MiB")
    print(f"Used GPU Memory: {used_memory} MiB")
    print(f"Free GPU Memory: {free_memory} MiB")
    print(f"Memory Usage: {used_memory/total_memory*100:.1f}%\n")
    
    # Check for 'zombie' memory (used but not accounted for by processes)
    processes_sum = df['Memory Usage (MiB)'].sum() if not df.empty else 0
    zombie_memory = used_memory - processes_sum
    
    print(f"Memory accounted for by processes: {processes_sum} MiB")
    print(f"'Zombie' memory (used but not accounted for): {zombie_memory} MiB\n")
    
    # Print processes by memory usage (sorted)
    if not df.empty:
        print("Process Details (sorted by memory usage):")
        df_sorted = df.sort_values(by='Memory Usage (MiB)', ascending=False)
        
        # Format the output to be more readable
        pd.set_option('display.max_colwidth', 50)
        print(df_sorted.to_string(index=False))
    
    # Print recommendations
    print("\nRecommendations:")
    if zombie_memory > total_memory * 0.2:  # If zombie memory is more than 20% of total
        print("- High amount of 'zombie' memory detected. Consider restarting your GPU processes or rebooting.")
    
    if not df.empty:
        high_usage_procs = df[df['Memory Usage (MiB)'] > total_memory * 0.2]
        for _, proc in high_usage_procs.iterrows():
            print(f"- Process {proc['Process Name']} (PID: {proc['PID']}) is using {proc['Memory Usage (MiB)']} MiB ({proc['Memory Usage (MiB)']/total_memory*100:.1f}% of total).")
    
    if free_memory < total_memory * 0.1:  # Less than 10% free
        print("- Low free memory. Consider closing unnecessary applications.")

if __name__ == "__main__":
    main()