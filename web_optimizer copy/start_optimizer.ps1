#!/usr/bin/env powershell

# Set execution policy for this session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# Add Node.js to PATH
$env:PATH = "C:\Program Files\nodejs;$env:PATH"

Write-Host "🚀 Starting MLB DFS Web Optimizer..." -ForegroundColor Green
Write-Host ""

# Check if Node.js is available
try {
    $nodeVersion = & node --version
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# Start backend server in new window
Write-Host "Starting backend server..." -ForegroundColor Yellow
Set-Location -Path "$PSScriptRoot\server"
Start-Process -FilePath "node" -ArgumentList "index.js" -WindowStyle Normal

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start frontend server
Write-Host "Starting frontend server..." -ForegroundColor Yellow
Set-Location -Path "$PSScriptRoot\client"

# Check if react-scripts is available
if (-not (Test-Path "node_modules\.bin\react-scripts.cmd")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    & npm.cmd install
}

Write-Host ""
Write-Host "🌐 Opening web optimizer in your browser..." -ForegroundColor Green
Write-Host "Backend: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""

# Start the React development server
& npm.cmd start
