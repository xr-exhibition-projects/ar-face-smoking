#define MyAppName        "VisoMaster"
#define MyAppVersion     "0.1.1"
#define MyAppPublisher   "VisoMaster Team"
#define MyAppURL         "https://github.com/visomaster/VisoMaster"
#define MyDistDir        "..\\dist\\VisoMaster"

[Setup]
AppId={{C2F9376C-20F2-4E25-9B1D-1C2BF0F451B0}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=VisoMaster_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\VisoMaster.exe
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#MyDistDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\VisoMaster.exe"
Name: "{group}\Start Portable Mode"; Filename: "{app}\Start_Portable.bat"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\VisoMaster.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VisoMaster.exe"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

