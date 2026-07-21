; Inno Setup 6.3+ installer for live-captions.
; (6.3 is required: ArchitecturesAllowed=x64os below was renamed from "x64" in 6.3.)
; Build the PyInstaller bundle first (packaging/livecaptions.spec -> dist/LiveCaptions),
; then compile this with ISCC:  iscc packaging\livecaptions.iss
;
; Design decisions (from the M7 investigation):
;  - PER-USER install (no admin). The app writes models + the transcripts DB to
;    %LOCALAPPDATA%\live-captions, which is per-user; a per-machine install in
;    Program Files could never clean another user's data on uninstall anyway.
;  - Uninstall PRESERVES the user's saved transcripts and config by default, and
;    offers to remove the downloaded models (potentially GBs). It never touches
;    anything outside our own %LOCALAPPDATA%\live-captions tree.

#define AppName "Live Captions"
#define AppVersion "0.4.3"
#define AppPublisher "live-captions"
#define AppExeCli "livecaptions.exe"
#define AppExeGui "livecaptions-overlay.exe"
; The patch build overrides this with /DSourceDir=..\dist\LiveCaptions-patch, so
; only define it when the command line didn't.
#ifndef SourceDir
  #define SourceDir "..\dist\LiveCaptions"
#endif

[Setup]
; A stable AppId keeps upgrades in place across versions. Keep this GUID fixed.
AppId={{7C1D8F5A-3B2E-4C9A-9E1D-4F6A2B8C0D11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
; Per-user: lowest privileges, install under the user's local Programs dir.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={autopf}\LiveCaptions
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; 64-bit only. x64os (not the deprecated "x64") blocks Arm64 cleanly â€” there are
; no Arm64 CUDA wheels and emulated CPU inference won't be real-time.
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
CloseApplications=yes
RestartApplications=no
OutputDir=..\Output
; PATCH build: same AppId, so it upgrades an existing install in place â€” but its
; SourceDir holds ONLY the two exes (+ internal.sha256), not the ~800 MB _internal
; tree. Inno's uninstall log is cumulative across same-AppId installs, so the
; _internal files the FULL installer logged stay tracked and are still removed on
; uninstall. Only offered by the updater when internal.sha256 proves the libraries
; are unchanged; a fresh install always uses the full installer.
#ifdef PatchBuild
OutputBaseFilename=LiveCaptions-Patch-{#AppVersion}
#else
OutputBaseFilename=LiveCaptions-Setup-{#AppVersion}
#endif
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeGui}
; The payload is >1 GB â€” allow room and show a realistic estimate.
DiskSpanning=no
LicenseFile=..\packaging\licenses\NOTICE.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked
Name: "addtopath"; Description: "Add the &command-line tool (livecaptions) to PATH"; GroupDescription: "Command line:"; Flags: unchecked

[Files]
; The whole PyInstaller --onedir output. ignoreversion on everything so a native
; DLL DOWNGRADE across versions still overwrites (Inno keeps same/newer by default).
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#ifndef PatchBuild
; Visual C++ runtime: ctranslate2.dll / onnxruntime.dll statically import
; msvcp140 / vcruntime140; installing it removes a whole class of "missing DLL"
; failures on a fresh box. Place vc_redist.x64.exe next to this .iss before compiling.
; A patch only runs over an existing full install, where the runtime is already in.
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
#endif

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeGui}"
Name: "{group}\{#AppName} (command line)"; Filename: "{app}\{#AppExeCli}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeGui}"; Tasks: desktopicon

[Run]
#ifndef PatchBuild
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing Visual C++ runtime..."; Flags: waituntilterminated
#endif
Filename: "{app}\{#AppExeGui}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
; The in-app updater installs silently and passes /RELAUNCH=1 so the app comes
; back up on its own. A plain silent install (winget, scripted) must NOT launch.
Filename: "{app}\{#AppExeGui}"; Flags: nowait; Check: ShouldRelaunch

[Code]
const
  EnvKey = 'Environment';

function ShouldRelaunch: Boolean;
begin
  Result := ExpandConstant('{param:RELAUNCH|0}') = '1';
end;

{ ---- PATH add/remove (HKCU): careful string surgery, never clobber the user's PATH ---- }
function PathList: string;
begin
  if not RegQueryStringValue(HKCU, EnvKey, 'Path', Result) then
    Result := '';
end;

function PathContains(const Dir: string): Boolean;
var
  Hay, Needle: string;
begin
  Hay := ';' + Uppercase(PathList) + ';';
  Needle := ';' + Uppercase(Dir) + ';';
  Result := Pos(Needle, Hay) > 0;
end;

procedure AddToPath(const Dir: string);
var
  Cur: string;
begin
  if PathContains(Dir) then exit;
  Cur := PathList;
  if (Cur <> '') and (Cur[Length(Cur)] <> ';') then Cur := Cur + ';';
  RegWriteExpandStringValue(HKCU, EnvKey, 'Path', Cur + Dir);
end;

procedure RemoveFromPath(const Dir: string);
var
  Cur, Rebuilt, Part: string;
  P: Integer;
begin
  Cur := PathList;
  if Cur = '' then exit;
  Rebuilt := '';
  Cur := Cur + ';';
  repeat
    P := Pos(';', Cur);
    Part := Copy(Cur, 1, P - 1);
    Cur := Copy(Cur, P + 1, Length(Cur));
    if (Part <> '') and (Uppercase(Part) <> Uppercase(Dir)) then
    begin
      if Rebuilt <> '' then Rebuilt := Rebuilt + ';';
      Rebuilt := Rebuilt + Part;
    end;
  until Cur = '';
  RegWriteExpandStringValue(HKCU, EnvKey, 'Path', Rebuilt);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    if WizardIsTaskSelected('addtopath') then
      AddToPath(ExpandConstant('{app}'));
end;

{ ---- Uninstall: preserve user data by default, offer to drop models ---- }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir, ModelsDir, HfDir: string;
begin
  if CurUninstallStep <> usUninstall then exit;

  RemoveFromPath(ExpandConstant('{app}'));

  DataDir := ExpandConstant('{localappdata}\live-captions');
  ModelsDir := DataDir + '\models';
  HfDir := DataDir + '\hf';

  { transcripts.db + config are the user's data â€” never delete them.
    Models are big and re-downloadable â€” offer to remove. Silent uninstall
    (e.g. winget) keeps everything. }
  if not UninstallSilent then
  begin
    if DirExists(ModelsDir) or DirExists(HfDir) then
      if MsgBox('Also delete the downloaded speech models? They can be several GB '
              + 'and will be re-downloaded if you reinstall.'
              + #13#10#13#10 + 'Your saved transcripts and settings are kept either way.',
              mbConfirmation, MB_YESNO) = IDYES then
      begin
        DelTree(ModelsDir, True, True, True);
        DelTree(HfDir, True, True, True);
      end;
  end;
end;
