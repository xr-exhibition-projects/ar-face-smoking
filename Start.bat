
call conda activate env50
REM Оптимизация: проверяем, нужно ли конвертировать UI файлы
REM Конвертируем только если UI файл новее сгенерированного Python файла
if not exist "app\ui\core\main_window.py" (
    call app/ui/core/convert_ui_to_py.bat
) else (
    for %%F in ("app\ui\core\MainWindow.ui") do set UI_TIME=%%~tF
    for %%F in ("app\ui\core\main_window.py") do set PY_TIME=%%~tF
    if "!UI_TIME!" gtr "!PY_TIME!" (
        call app/ui/core/convert_ui_to_py.bat
    )
)
SET APP_ROOT=%~dp0
SET APP_ROOT=%APP_ROOT:~0,-1%
SET DEPENDENCIES=%APP_ROOT%\dependencies
echo %DEPENDENCIES%
SET PATH=%DEPENDENCIES%;%PATH%
python main.py
pause