
REM Название conda окружения можно задать несколькими способами:
REM 1. Через переменную окружения VISO_CONDA_ENV
REM 2. Через файл conda_env.txt в корне проекта (первая строка)
REM 3. Если ничего не задано, используется значение по умолчанию "visomaster"

if "%VISO_CONDA_ENV%"=="" (
    REM Пробуем прочитать из файла conda_env.txt
    if exist "conda_env.txt" (
        for /f "tokens=*" %%a in (conda_env.txt) do (
            set VISO_CONDA_ENV=%%a
            goto :found_env
        )
    )
    REM Если файл не найден, используем значение по умолчанию
    set VISO_CONDA_ENV=visomaster
)
:found_env
echo Используется conda окружение: %VISO_CONDA_ENV%
call conda activate %VISO_CONDA_ENV%
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