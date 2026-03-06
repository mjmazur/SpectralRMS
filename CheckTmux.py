import os, subprocess, time

if __name__ == "__main__":
	start_command = "~/Desktop/StartTmuxSessions.sh"
	try:
		result = subprocess.run(['tmux', 'list-windows'], capture_output=True, text=True, check=True)
		output_lines = result.stdout.strip().splitlines()
	except:
		output_lines = ['Dead']
	
	if output_lines[0] == 'Dead':
		os.system(start_command)