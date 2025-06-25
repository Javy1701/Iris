@echo off
REM ############################################################################
REM #                                                                          #
REM #         Iris Development Server Startup Script for Windows               #
REM #                                                                          #
REM #   This script automates the process of activating the Python virtual     #
REM #   environment (venv) and starting the Uvicorn server for a FastAPI app.  #
REM #                                                                          #
REM #   Place this file in the root directory of your project.                 #
REM #                                                                          #
REM ############################################################################

REM --- Set the window title for clarity ---
title Iris Server

REM --- Define project variables (Customize these if needed) ---
SET VENV_PATH=.venv
SET MAIN_APP_FILE=app.main:app

echo -------------------------------------
echo  Iris Server Starting...
echo -------------------------------------

REM --- Check if the virtual environment exists ---
IF NOT EXIST "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at '%VENV_PATH%'.
    echo Please make sure you have created a venv using 'python -m venv venv'.
    pause
    exit /b
)

echo [INFO] Activating the virtual environment...
REM --- Activate the virtual environment. 'CALL' is important here! ---
CALL %VENV_PATH%\Scripts\activate.bat

echo [INFO] Starting the Uvicorn server...
echo [INFO] Your app will be available at http://127.0.0.1:8000
echo [INFO] Press CTRL+C to stop the server.
echo.

REM --- Start the FastAPI server using Uvicorn ---
REM The --reload flag automatically restarts the server when you save code changes.
uvicorn %MAIN_APP_FILE% --reload --host 0.0.0.0 --port 8000

echo [INFO] Server has been stopped.
pause

