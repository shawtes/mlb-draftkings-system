#!/bin/bash

echo "======================================"
echo "GSU Server - Auto Run Script"
echo "======================================"
echo ""
echo "This will upload files and show you the SSH command."
echo ""

SERVER="stesfaye4@snowball.cs.gsu.edu"

echo "Step 1: Uploading files..."
scp part1a.c part1a_answer.txt part1b_answer.txt part2.c part2_answer.txt part3.c part4.c RUN_ON_SERVER.sh $SERVER:~/

echo ""
echo "Step 2: Connecting to server..."
echo ""
echo "After connecting, run these commands:"
echo ""
echo "  chmod +x RUN_ON_SERVER.sh"
echo "  bash RUN_ON_SERVER.sh > lab_output.txt"
echo "  cat lab_output.txt"
echo ""
echo "⚠️  Then change your password: passwd"
echo ""
echo "Connecting now..."
echo ""

ssh $SERVER





