/* w3launch.exe: start a program suspended, load a DLL into it, then let it run.
 *
 *   w3launch.exe <dll> <exe> [args...]
 *
 * The child's command line is the exe path (quoted) followed by the remaining arguments,
 * re-quoted. The launcher waits for the child and returns its exit code.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

static void append_arg(wchar_t *cmd, size_t cap, const wchar_t *arg) {
    size_t len = wcslen(cmd);
    int quote = wcspbrk(arg, L" \t\"") != NULL || !*arg;
    if (len && len < cap - 1)
        cmd[len++] = L' ';
    if (quote && len < cap - 1)
        cmd[len++] = L'"';
    for (const wchar_t *p = arg; *p && len < cap - 2; p++) {
        if (*p == L'"' && len < cap - 3)
            cmd[len++] = L'\\';
        cmd[len++] = *p;
    }
    if (quote && len < cap - 1)
        cmd[len++] = L'"';
    cmd[len] = 0;
}

int wmain(int argc, wchar_t **argv) {
    if (argc < 3) {
        fwprintf(stderr, L"usage: w3launch.exe <dll> <exe> [args...]\n");
        return 2;
    }
    wchar_t dll[MAX_PATH], exe[MAX_PATH], dir[MAX_PATH];
    if (!GetFullPathNameW(argv[1], MAX_PATH, dll, NULL) || !GetFullPathNameW(argv[2], MAX_PATH, exe, NULL)) {
        fwprintf(stderr, L"w3launch: bad path\n");
        return 2;
    }
    wcscpy(dir, exe);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash)
        *slash = 0;

    static wchar_t cmd[32768];
    cmd[0] = 0;
    append_arg(cmd, 32768, exe);
    for (int i = 3; i < argc; i++)
        append_arg(cmd, 32768, argv[i]);

    STARTUPINFOW si = {sizeof si};
    PROCESS_INFORMATION pi;
    if (!CreateProcessW(exe, cmd, NULL, NULL, FALSE, CREATE_SUSPENDED, NULL, dir, &si, &pi)) {
        fwprintf(stderr, L"w3launch: CreateProcess failed (%lu)\n", GetLastError());
        return 3;
    }

    size_t bytes = (wcslen(dll) + 1) * sizeof(wchar_t);
    void *remote = VirtualAllocEx(pi.hProcess, NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    LPTHREAD_START_ROUTINE load =
        (LPTHREAD_START_ROUTINE)GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW");
    HANDLE th = NULL;
    if (remote && WriteProcessMemory(pi.hProcess, remote, dll, bytes, NULL))
        th = CreateRemoteThread(pi.hProcess, NULL, 0, load, remote, 0, NULL);
    DWORD loaded = 0;
    if (th) {
        WaitForSingleObject(th, 30000);
        GetExitCodeThread(th, &loaded);
        CloseHandle(th);
    }
    if (!loaded) {
        fwprintf(stderr, L"w3launch: injecting %ls failed (%lu)\n", dll, GetLastError());
        TerminateProcess(pi.hProcess, 4);
        return 4;
    }
    VirtualFreeEx(pi.hProcess, remote, 0, MEM_RELEASE);

    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    return (int)code;
}
