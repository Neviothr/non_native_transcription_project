@echo off
cd /d "%~dp0"
where pyw >nul 2>&1
if %errorlevel%==0 (
    start "" pyw -3.14 setup_gui.py
    exit /b
)
where pythonw >nul 2>&1
if %errorlevel%==0 (
    start "" pythonw setup_gui.py
    exit /b
)
msg * Python 3.14.6 was not found. Install it from python.org and include Tcl/Tk.




