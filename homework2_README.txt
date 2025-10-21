CSC 3320 - Homework 2 Instructions
===================================

FILES INCLUDED:
- homework2.c       (C program)
- homework2.bash    (Shell script)
- homework2_README.txt (This file)

IMPORTANT: Before running, update your user ID in homework2.c
-----------------------------------------------------------
1. Find your user ID by running: id -u
2. Open homework2.c and replace "1001" on line 29 with your actual UID
3. Replace "Sineshaw Mesfin Tesfaye" on line 32 with your actual name

COMPILING THE C PROGRAM:
------------------------
gcc homework2.c -o homework2

RUNNING THE C PROGRAM:
----------------------
./homework2

MAKING THE SHELL SCRIPT EXECUTABLE:
------------------------------------
chmod +x homework2.bash

RUNNING THE SHELL SCRIPT:
--------------------------
./homework2.bash

CONNECTING TO GSU SERVER:
--------------------------
ssh stesfaye4@student.cs.gsu.edu

(Then enter your password when prompted)

TESTING ON GSU SERVER:
----------------------
1. Connect to the server using ssh command above
2. Upload your files using scp or create them with nano/vim
3. Compile and run as shown above
4. Capture the output for your log file

TO CAPTURE OUTPUT FOR LOG FILE:
--------------------------------
You can redirect output to a file:
./homework2 > c_program_log.txt
./homework2.bash > shell_script_log.txt

Or use the script command to record your entire session:
script session_log.txt
# run your programs here
exit

NOTES:
------
- The /etc/passwd file on the GSU server may not show student accounts
- This is expected behavior - the assignment instructions mention this
- Your grep command should still run, it just may return no results
- The process lists (ps output) will likely differ between runs due to timing

