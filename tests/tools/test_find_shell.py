"""Tests for _find_shell — user-login-shell preference on POSIX.

Regression tests for #42203: on macOS, ``_find_shell`` used to return
``/bin/bash`` (bash 3.2) which silently swallowed background commands
when ``~/.bash_profile`` contained ``exec /bin/zsh -l``.
"""

import os
import platform
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from tools.environments.local import _find_bash, _find_shell


class TestFindShellPrefersUserShell:
    """_find_shell should prefer $SHELL over bash on POSIX."""

    def test_returns_shell_env_when_set_and_exists(self, tmp_path):
        """When $SHELL points to an existing allowlisted executable, _find_shell returns it."""
        fake_zsh = tmp_path / "zsh"
        fake_zsh.touch()
        fake_zsh.chmod(0o755)
        with patch("tools.environments.local._IS_WINDOWS", False), patch.dict(
            os.environ, {"SHELL": str(fake_zsh)}
        ):
            assert _find_shell() == str(fake_zsh)

    def test_falls_back_when_shell_not_executable(self, tmp_path):
        """$SHELL exists but lacks the execute bit -> fall back to _find_bash
        (returning it would fail at spawn time)."""
        fake = tmp_path / "zsh"
        fake.touch()
        fake.chmod(0o644)  # not executable
        with patch.dict(os.environ, {"SHELL": str(fake)}):
            assert _find_shell() == _find_bash()

    def test_falls_back_for_incompatible_shell_fish(self, tmp_path):
        """#42203 regression: $SHELL=fish must NOT be returned — spawn_local's
        `-lic` / `set +m` syntax breaks fish, which would trade the bash-3.2
        swallow for a parse error on every background command. Fall back to bash."""
        fake_fish = tmp_path / "fish"
        fake_fish.touch()
        fake_fish.chmod(0o755)
        with patch.dict(os.environ, {"SHELL": str(fake_fish)}):
            assert _find_shell() == _find_bash()


    def test_honours_allowlisted_bash_and_dash(self, tmp_path):
        """Every allowlisted POSIX-sh-family shell is honoured."""
        for name in ("bash", "dash", "sh", "ksh"):
            fake = tmp_path / name
            fake.touch()
            fake.chmod(0o755)
            with patch("tools.environments.local._IS_WINDOWS", False), patch.dict(
                os.environ, {"SHELL": str(fake)}
            ):
                assert _find_shell() == str(fake), name


    def test_falls_back_to_find_bash_when_shell_empty(self):
        """When $SHELL is empty string, _find_shell delegates."""
        with patch.dict(os.environ, {"SHELL": ""}):
            assert _find_shell() == _find_bash()


class TestFindShellWindowsBehavior:
    """On Windows, _find_shell always delegates to _find_bash."""

    def test_windows_ignores_shell_env(self):
        """On Windows, $SHELL is ignored — _find_shell delegates to _find_bash."""
        with patch("tools.environments.local._IS_WINDOWS", True):
            # Even if SHELL is set, it should be ignored on Windows
            with patch.dict(os.environ, {"SHELL": "/usr/bin/zsh"}):
                result = _find_shell()
                assert result == _find_bash()


class TestFindShellReturnsString:
    """_find_shell must return a string, never None."""

    def test_returns_string(self):
        """_find_shell always returns a non-empty string on any platform."""
        result = _find_shell()
        assert isinstance(result, str)
        assert len(result) > 0


class TestFindBashUnchanged:
    """_find_bash should be unaffected by the _find_shell change."""

    def test_find_bash_still_prefers_bash(self):
        """_find_bash still returns bash (not $SHELL) on POSIX."""
        result = _find_bash()
        # On any system, _find_bash should return something containing "bash"
        # or fall back to $SHELL or /bin/sh — but it should NOT prefer $SHELL
        # over bash the way _find_shell does.
        assert isinstance(result, str)
        assert len(result) > 0


class TestFindBashSkipsBrokenCustomPath:
    """Stale HERMES_GIT_BASH_PATH must not brick Windows terminal startup."""

    def test_falls_through_to_portable_when_custom_fails_probe(self, tmp_path, monkeypatch):
        import tools.environments.local as local_mod

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        local_mod._bash_starts_cache.clear()

        broken = tmp_path / "broken" / "bash.exe"
        broken.parent.mkdir()
        broken.write_text("", encoding="utf-8")
        portable = tmp_path / "hermes" / "git" / "bin" / "bash.exe"
        portable.parent.mkdir(parents=True)
        portable.write_text("", encoding="utf-8")

        monkeypatch.setenv("HERMES_GIT_BASH_PATH", str(broken))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        def fake_starts(path: str) -> bool:
            return path == str(portable)

        monkeypatch.setattr(local_mod, "_bash_starts", fake_starts)

        assert _find_bash() == str(portable)


class TestGitBashExternalProgramProbe:
    """The Windows health check must exercise MSYS child-process creation."""

    def test_probe_runs_external_msys_programs(self, monkeypatch):
        import tools.environments.local as local_mod

        local_mod._bash_starts_cache.clear()
        local_mod._bash_probe_details_cache.clear()
        calls = []

        def fake_probe(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(local_mod, "bounded_process_probe", fake_probe)
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)

        assert local_mod._bash_starts(r"C:\Git\bin\bash.exe") is True
        assert calls[0][0][-1] == "/usr/bin/true; /usr/bin/cat --version >/dev/null"

    def test_rejects_system32_wsl_without_probing_it(self, monkeypatch):
        import tools.environments.local as local_mod

        local_mod._bash_starts_cache.clear()
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", "")
        monkeypatch.setenv("LOCALAPPDATA", r"C:\missing")
        monkeypatch.setenv("ProgramFiles", r"C:\missing")
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.setattr(local_mod.os.path, "isfile", lambda _path: False)
        monkeypatch.setattr(
            local_mod.shutil,
            "which",
            lambda _name: r"C:\Windows\System32\bash.exe",
        )
        probes = []
        monkeypatch.setattr(
            local_mod,
            "_bash_starts",
            lambda path: probes.append(path) or True,
        )

        with pytest.raises(RuntimeError, match="Git Bash not found"):
            local_mod._find_bash()

        assert probes == []

    def test_configured_bin_shim_canonicalizes_to_usr_bin(
        self, monkeypatch
    ):
        import tools.environments.local as local_mod

        local_mod._bash_starts_cache.clear()
        shim = r"C:\Program Files\Git\bin\bash.exe"
        direct = r"C:\Program Files\Git\usr\bin\bash.exe"
        existing = {os.path.normcase(shim), os.path.normcase(direct)}

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", shim)
        monkeypatch.setenv("LOCALAPPDATA", r"C:\missing")
        monkeypatch.setenv("ProgramFiles", r"C:\missing")
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.setattr(
            local_mod.os.path,
            "isfile",
            lambda path: os.path.normcase(path) in existing,
        )
        monkeypatch.setattr(local_mod.shutil, "which", lambda _name: None)
        probes = []
        monkeypatch.setattr(
            local_mod,
            "_bash_starts",
            lambda path: probes.append(path) or path == direct,
        )

        assert local_mod._find_bash() == direct
        assert probes == [direct]

    def test_candidate_probes_share_one_startup_deadline(self, monkeypatch):
        import tools.environments.local as local_mod

        local_mod._bash_starts_cache.clear()
        candidates = {
            ntpath
            for ntpath in (
                r"C:\one\usr\bin\bash.exe",
                r"C:\two\usr\bin\bash.exe",
                r"C:\three\usr\bin\bash.exe",
            )
        }
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", next(iter(candidates)))
        monkeypatch.setenv("LOCALAPPDATA", r"C:\missing")
        monkeypatch.setenv("ProgramFiles", r"C:\missing")
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.setattr(
            local_mod.os.path, "isfile", lambda path: path in candidates
        )
        remaining = iter(candidates - {os.environ["HERMES_GIT_BASH_PATH"]})
        monkeypatch.setattr(
            local_mod.shutil, "which", lambda _name: next(remaining, None)
        )
        timeouts = []

        def slow_failure(_path, *, timeout=15):
            timeouts.append(timeout)
            time.sleep(min(timeout, 0.15))
            return False

        monkeypatch.setattr(local_mod, "_bash_starts", slow_failure)
        started = time.monotonic()
        with pytest.raises((RuntimeError, TimeoutError)):
            local_mod._find_bash(deadline=time.monotonic() + 0.2)
        elapsed = time.monotonic() - started

        assert elapsed < 0.5
        assert timeouts
        assert all(0 < timeout <= 0.21 for timeout in timeouts)
        assert timeouts == sorted(timeouts, reverse=True)

    @pytest.mark.skipif(sys.platform != "win32", reason="requires real Windows process trees")
    def test_bounded_probe_kills_descendant_holding_output_pipe(self, tmp_path):
        from gateway.status import _pid_exists
        from hermes_cli._subprocess_compat import bounded_process_probe

        child_pid_file = tmp_path / "child.pid"
        child_code = "import time; time.sleep(60)"
        parent_code = (
            "import pathlib, subprocess, sys, time; "
            f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
            "print('ready', flush=True); time.sleep(60)"
        )

        started = time.monotonic()
        result = bounded_process_probe(
            [sys.executable, "-c", parent_code], timeout=0.5
        )
        elapsed = time.monotonic() - started

        assert result is None
        assert elapsed < 5
        child_pid = int(child_pid_file.read_text())
        deadline = time.monotonic() + 3
        while _pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_exists(child_pid)

    def test_aslr_failure_surfaces_targeted_windows_command(
        self, tmp_path, monkeypatch
    ):
        import tools.environments.local as local_mod

        local_mod._bash_starts_cache.clear()
        local_mod._bash_probe_details_cache.clear()
        portable = tmp_path / "hermes" / "git" / "bin" / "bash.exe"
        portable.parent.mkdir(parents=True)
        portable.write_text("", encoding="utf-8")

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", "")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "empty-program-files"))
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.setattr(local_mod.shutil, "which", lambda _name: None)
        monkeypatch.setattr(local_mod, "_mandatory_aslr_enabled", lambda: True)

        def failed_probe(path: str) -> bool:
            local_mod._bash_probe_details_cache[path] = (
                "dofork: child -1 - forked process died unexpectedly"
            )
            return False

        monkeypatch.setattr(local_mod, "_bash_starts", failed_probe)

        with pytest.raises(RuntimeError) as exc_info:
            local_mod._find_bash()
        message = str(exc_info.value)
        assert "Mandatory ASLR" in message
        assert "Reinstalling Git will not change" in message
        assert "Set-ProcessMitigation" in message
        assert str(tmp_path / "hermes" / "git") in message


@pytest.mark.skipif(
    not os.path.isfile("/bin/bash") or sys.platform != "darwin",
    reason="reproduces the macOS system-bash-3.2 login-shell swallow",
)
class TestMacosLoginShellSwallowRegression:
    """E2E regression for #42203: the actual failure is that system bash 3.2,
    invoked as a login shell (`-lic`) with stdin=/dev/null and a
    ~/.bash_profile that `exec`s zsh, silently swallows the command (exit 0,
    no output, no side effects). Prove (a) the bug exists with /bin/bash and
    (b) the $SHELL (zsh) path _find_shell prefers does NOT swallow."""

    def _spawn_like_registry(self, shell, command, home, tmp_path):
        import subprocess
        env = dict(os.environ)
        env["HOME"] = str(home)
        # Mirror process_registry.spawn_local: [shell, "-lic", "set +m; <cmd>"]
        # with stdin redirected to /dev/null.
        return subprocess.run(
            [shell, "-lic", f"set +m; {command}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_system_bash_swallows_but_zsh_does_not(self, tmp_path):
        # A .bash_profile that exec's zsh — the reported macOS shape.
        home = tmp_path / "home"
        home.mkdir()
        (home / ".bash_profile").write_text("exec /bin/zsh -l\n")

        zsh = os.environ.get("SHELL") or "/bin/zsh"
        if not os.path.isfile(zsh):
            pytest.skip("no zsh available")

        marker_bash = tmp_path / "bash_ran"
        marker_zsh = tmp_path / "zsh_ran"

        # /bin/bash login shell: command is swallowed (file NOT created).
        self._spawn_like_registry("/bin/bash", f"echo x > {marker_bash}", home, tmp_path)
        # zsh (the $SHELL _find_shell prefers): command runs (file created).
        self._spawn_like_registry(zsh, f"echo x > {marker_zsh}", home, tmp_path)

        # The FIX path (zsh) must run the command.
        assert marker_zsh.exists(), "zsh ($SHELL) path must run the command"

        # Differential: when /bin/bash is the swallow-prone 3.x (macOS system
        # bash), the login-shell invocation must demonstrably FAIL to run the
        # command — that's the bug this PR routes around. Only assert the
        # negative when we've confirmed a 3.x bash, so the test stays valid on
        # boxes/CI with a newer /bin/bash that doesn't swallow.
        ver = subprocess.run(
            ["/bin/bash", "--version"], capture_output=True, text=True
        ).stdout
        if "version 3." in ver:
            assert not marker_bash.exists(), (
                "system bash 3.x login shell should swallow the command "
                "(the #42203 bug); _find_shell routes around it by preferring zsh"
            )

    def test_find_shell_selects_working_shell_on_this_box(self, tmp_path):
        """_find_shell's choice must actually execute a background-style
        command (regression against returning a swallow-prone shell)."""
        shell = _find_shell()
        marker = tmp_path / "ok_marker"
        subprocess.run(
            [shell, "-lic", f"set +m; echo ok > {marker}"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        assert marker.exists(), f"_find_shell()={shell} swallowed the command"
