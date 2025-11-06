@echo off
echo Starting Stock Price Prediction App...
echo.
echo Starting Flask API backend on port 5000...
echo Starting HTML frontend on port 8080...
echo.
echo The application will open in your default browser automatically.
echo Press Ctrl+C in either window to stop the application.
echo.

REM Start Flask API backend in background
start "Flask API Backend" cmd /k "python app.py serve"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start simple HTTP server for frontend
start "HTML Frontend" cmd /k "python -m http.server 8080"

REM Wait a moment for frontend server to start
timeout /t 2 /nobreak >nul

REM Open the application in default browser
start http://localhost:8080

echo.
echo Application started successfully!
echo Backend API: http://localhost:5000
echo Frontend: http://localhost:8080
echo.
echo Close both command windows to stop the application.
pause