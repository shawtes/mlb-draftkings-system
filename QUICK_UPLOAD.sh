#!/bin/bash

echo "SSH Server Upload Script"
echo "========================="
echo ""
read -p "Enter your username: " username
read -p "Enter server address (e.g., cs.school.edu): " server
echo ""

echo "Uploading files to $username@$server..."
echo ""

scp part1a.c part1a_answer.txt part1b_answer.txt part2.c part2_answer.txt part3.c part4.c $username@$server:~/

echo ""
echo "Files uploaded!"
echo ""
echo "Now connect with:"
echo "ssh $username@$server"
echo ""
echo "Then follow SSH_SERVER_INSTRUCTIONS.txt"





