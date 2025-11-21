#!/bin/bash

echo "======================================"
echo "GSU Server Upload Script"
echo "======================================"
echo ""

SERVER="stesfaye4@snowball.cs.gsu.edu"

echo "Uploading lab files to $SERVER..."
echo ""

scp part1a.c part1a_answer.txt part1b_answer.txt part2.c part2_answer.txt part3.c part4.c RUN_ON_SERVER.sh $SERVER:~/

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Files uploaded successfully!"
    echo ""
    echo "Now connect with:"
    echo "ssh $SERVER"
    echo ""
    echo "Then run:"
    echo "bash RUN_ON_SERVER.sh"
    echo ""
    echo "⚠️  IMPORTANT: Change your password after this!"
    echo "Run: passwd"
else
    echo ""
    echo "❌ Upload failed. Check your connection."
fi









