#!/bin/bash

# Set the source folder
source_folder="/opt/habanalabs/qual/gaudi2/bin"

# Temporary directory to store output files
output_temp_dir="/tmp/qual_output"

# Create the temporary output directory if it doesn't exist
mkdir -p "$output_temp_dir"

# List of all commands
all_commands=(
    "./hl_qual -c all -dis_mon -rmod serial -mb -b -pciOnly -gen gen4 -gaudi2"
    "./hl_qual -dis_mon -gaudi2 -c all -e2e_concurrency -disable_ports 8,22,23 -rmod parallel -enable_ports_check int"
    "./hl_qual -dis_mon -gaudi2 -c all -f2 -rmod parallel -l extreme -t 120 -serdes -enable_ports_check int"
    "hl-smi -q | grep SPI"
    "sudo apt list --installed | grep habana"
)

# Renamed command names
command_names=(
    "memory_bw_pci"
    "e2e"
    "f2e_using_serdes"
    "hl-smi-spi"
    "apt-list-habana"
)

# Print script usage information
print_help() {
    echo "Usage: $0 [command_name | all]"
    echo "       $0 --help"
    echo ""
    echo "Options:"
    echo "  command_name    Specify the command name to run a specific command."
    echo "                  Available command names: ${command_names[*]}"
    echo "                  Use 'all' to run all commands."
    echo "  --help          Display this help message."
    echo ""
}

# Get the parameter (if provided)
specified_command="$1"

# Handle --help option
if [[ "$specified_command" == "--help" ]]; then
    print_help
    exit 0
fi

# If a specific command is provided as a parameter, only run that command
if [[ -n "$specified_command" ]]; then
    if [[ "$specified_command" == "all" ]]; then
        for cmd_name in "${command_names[@]}"; do
            $0 "$cmd_name"
        done
    else
        cmd_index=-1
        for index in "${!command_names[@]}"; do
            if [[ "$specified_command" == "${command_names[$index]}" ]]; then
                cmd_index=$index
                break
            fi
        done

        if [[ $cmd_index -eq -1 ]]; then
            echo "Specified command not found: $specified_command"
            exit 1
        fi

        cmd="${all_commands[$cmd_index]}"
        cmd_name="${command_names[$cmd_index]}"
        output_file="$output_temp_dir/${cmd_name}.txt"

        echo "Running: $cmd"
        (
            if [[ "$cmd" == *"sudo"* ]]; then
                bash -c "$cmd" > "$output_file" 2>&1
            else
                cd "$source_folder"
                bash -c "$cmd" > "$output_file" 2>&1
            fi
        )

        echo "Output saved to: $output_file"
    fi

# If no parameter provided, run all commands
else
    print_help
    exit 1
fi

