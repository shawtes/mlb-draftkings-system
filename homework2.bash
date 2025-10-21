#!/bin/bash

# Starting message
echo "Starting shell script"
echo ""

# Get and display process ID
echo "Process ID: $$"

# Get and display parent process ID
echo "Parent Process ID: $PPID"

# Get and display group ID
# GID is not defined in all shells (like bash), so we define it
GID=`id -g`
echo "Group ID: $GID"
echo ""

# Welcome message with username
echo "Welcome $USER"

# Search for username in /etc/passwd
echo "Searching for user in /etc/passwd:"
grep $USER /etc/passwd
echo ""

# Get first list of processes
echo "First process list:"
ps
echo ""

# Sleep for 3 seconds
echo "Sleeping for 3 seconds..."
echo ""
sleep 3

# Get second list of processes
echo "Second process list:"
ps
echo ""

echo "Note: The process lists may differ slightly due to timing."
echo ""

# Ending message
echo "Ending shell script"

