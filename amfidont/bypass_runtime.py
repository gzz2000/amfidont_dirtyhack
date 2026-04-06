from pprint import pprint
import re
import subprocess
import struct
import threading
from collections import deque
from typing import Dict, List, Optional, Set, Tuple, Union

from amfidont.config_store import config_modified_time_state, load_persistent_config
from amfidont.lldb_importer import lldb


AMFID_PATH = "/usr/libexec/amfid"
# Offsets recovered from -[AMFIPathValidator_macos ...] ivar table in
# AppleMobileFileIntegrity.framework (arm64e macOS build).
AMFI_VALIDATOR_VALIDATED_OFFSET = 16
AMFI_VALIDATOR_HAS_RESTRICTED_ENTITLEMENTS_OFFSET = 36
AMFI_VALIDATOR_HAS_ONLY_SOFT_RESTRICTED_ENTITLEMENTS_OFFSET = 37
AMFI_VALIDATOR_IS_SIGNED_OFFSET = 48
AMFI_VALIDATOR_IS_VALID_OFFSET = 49
AMFI_VALIDATOR_ARE_ENTITLEMENTS_VALIDATED_OFFSET = 50
AMFI_VALIDATOR_ERROR_OBJECT_OFFSET = 72
# One-time calibration results:
# validator_ptr --(offset chain)--> C-string path / 20-byte cdhash bytes.
PATH_POINTER_CHAIN: Optional[Tuple[int, ...]] = None
CDHASH_POINTER_CHAIN: Optional[Tuple[int, ...]] = None
CDHASH_BY_PATH_CACHE: Dict[str, str] = {}
ACTIVE_DEBUGGER: Optional[lldb.SBDebugger] = None
ACTIVE_VERBOSE = False
PATCHED_VALIDATE_CALLSITES: Set[int] = set()
REGS_BY_ARCH = {
    "arm64": ("x0", "x0", "x2"),
    "x86_64": ("rax", "rdi", "rdx"),
}
VALIDATOR_METHODS = {
    "-[AMFIPathValidator_macos validateWithError:]",
}


def registers_for_target(target: lldb.SBTarget) -> Tuple[str, str, str]:
    """
    Return register names for the target architecture.

    :param target: LLDB target being inspected.
    :return: A tuple of `(return_register, self_register, error_out_register)`.
    """
    triple = target.GetTriple().lower()
    for arch, registers in REGS_BY_ARCH.items():
        if arch in triple:
            return registers
    raise RuntimeError(f"Unsupported architecture triple: {triple}")


def parse_pointer(value: Optional[str]) -> int:
    """
    Parse a register/string value as an integer pointer.
    """
    if not value:
        return 0
    try:
        return int(value, 0)
    except (TypeError, ValueError):
        return 0


def clear_pointer_value(process: lldb.SBProcess, ptr: int) -> None:
    """
    Zero a pointer-sized slot in the debuggee when the address is valid.
    """
    if ptr == 0:
        return
    error = lldb.SBError()
    process.WriteMemory(ptr, b"\x00" * 8, error)


def write_register_value(
    target: lldb.SBTarget,
    thread: lldb.SBThread,
    frame: lldb.SBFrame,
    register_name: str,
    value: str,
) -> bool:
    """
    Best-effort register write: SBValue API first, then LLDB command fallback.
    """
    try:
        reg = frame.FindRegister(register_name)
        if reg and reg.IsValid() and reg.SetValueFromCString(value):
            return True
    except Exception:
        pass

    if ACTIVE_DEBUGGER is None:
        return False

    try:
        process = target.GetProcess()
        process.SetSelectedThread(thread)
        thread.SetSelectedFrame(0)
        command_result = lldb.SBCommandReturnObject()
        ACTIVE_DEBUGGER.GetCommandInterpreter().HandleCommand(
            f"register write {register_name} {value}",
            command_result,
        )
        succeeded = bool(command_result.Succeeded())
        if ACTIVE_VERBOSE and not succeeded:
            print(
                "register write fallback failed:",
                register_name,
                value,
                command_result.GetError().strip(),
            )
        return succeeded
    except Exception:
        return False


def patch_validate_callsite_isvalid_store(
    target: lldb.SBTarget,
    frame: lldb.SBFrame,
    verbose: bool = False,
) -> bool:
    """
    Patch caller instruction after validateWithError returns:
      mov x19, x0  ->  mov w19, #1
    This avoids relying on register writes that may be blocked in some LLDB setups.
    """
    process = target.GetProcess()
    lr_value = 0
    for reg_name in ("x30", "lr"):
        reg = frame.FindRegister(reg_name)
        if reg and reg.IsValid():
            lr_value = parse_pointer(reg.value)
            if lr_value:
                break
    if lr_value == 0:
        return False

    if lr_value in PATCHED_VALIDATE_CALLSITES:
        return True

    error = lldb.SBError()
    original = process.ReadMemory(lr_value, 4, error)
    if not error.Success() or len(original) != 4:
        return False

    # arm64: mov x19, x0  (AA 00 03 F3 little-endian: f3 03 00 aa)
    expected_original = b"\xf3\x03\x00\xaa"
    # arm64: mov w19, #1 (52 80 00 33 little-endian: 33 00 80 52)
    patched = b"\x33\x00\x80\x52"

    if original != expected_original and original != patched:
        if verbose:
            print(
                f"callsite patch skipped at {hex(lr_value)}: unexpected bytes {original.hex()}"
            )
        return False

    if original == patched:
        PATCHED_VALIDATE_CALLSITES.add(lr_value)
        return True

    write_error = lldb.SBError()
    process.WriteMemory(lr_value, patched, write_error)
    if not write_error.Success():
        if verbose:
            print(f"callsite patch failed at {hex(lr_value)}: {write_error.GetCString()}")
        return False

    verify_error = lldb.SBError()
    verify = process.ReadMemory(lr_value, 4, verify_error)
    if not verify_error.Success() or verify != patched:
        return False

    PATCHED_VALIDATE_CALLSITES.add(lr_value)
    if verbose:
        print(f"Patched validate callsite at {hex(lr_value)} to force isValid=1")
    return True


def force_return_success(
    target: lldb.SBTarget,
    thread: lldb.SBThread,
    frame: lldb.SBFrame,
    ret_reg: str,
    verbose: bool = False,
) -> bool:
    """
    Force an immediate successful return from the current frame.
    """
    def find_register(name_candidates: Tuple[str, ...]) -> Optional[lldb.SBValue]:
        for name in name_candidates:
            try:
                reg = frame.FindRegister(name)
                if reg and reg.IsValid():
                    return reg
            except Exception:
                pass
            try:
                reg = frame.reg[name]
                if reg and reg.IsValid():
                    return reg
            except Exception:
                pass
        return None

    def try_manual_arm64_return() -> bool:
        lr_reg = find_register(("x30", "lr"))
        if not lr_reg:
            if verbose:
                print("manual arm64 return failed: missing x30/lr register")
            return False
        lr_value = parse_pointer(lr_reg.value)
        if lr_value == 0:
            if verbose:
                print("manual arm64 return failed: lr/x30 is null")
            return False
        ret_ok = write_register_value(target, thread, frame, ret_reg, "1")
        pc_ok = write_register_value(target, thread, frame, "pc", hex(lr_value))
        if verbose and not (ret_ok and pc_ok):
            print(f"manual arm64 return register write failed: ret_ok={ret_ok} pc_ok={pc_ok}")
        return bool(ret_ok and pc_ok)

    triple = target.GetTriple().lower()
    has_return_from_frame = hasattr(thread, "ReturnFromFrame")
    if "arm64" in triple and try_manual_arm64_return():
        if verbose:
            print("Bypass strategy: manual arm64 pc<-lr early-success")
        return True

    if not has_return_from_frame:
        return False

    return_value = frame.EvaluateExpression("(BOOL)1")
    if not return_value or not return_value.GetError().Success():
        return_value = frame.EvaluateExpression("1")

    if not return_value or not return_value.GetError().Success():
        return False

    try:
        result = thread.ReturnFromFrame(frame, return_value)
    except Exception:
        return False
    if isinstance(result, bool):
        success = result
    elif hasattr(result, "Success"):
        success = bool(result.Success())
    else:
        success = True

    if success:
        try:
            write_register_value(target, thread, thread.frames[0], ret_reg, "1")
        except Exception:
            pass
    else:
        if verbose:
            print("ReturnFromFrame failed, will fall back to StepOut patching")

    return success


def validate_hook(
    target: lldb.SBTarget,
    thread: lldb.SBThread,
    paths: Set[str],
    cdhashes: Set[str],
    verbose: bool = False,
    allow_all: bool = False,
) -> None:
    """
    Patch validation result when the validator matches configured allow-rules.

    :param target: LLDB target attached to `amfid`.
    :param thread: The stopped thread at the validation breakpoint.
    :param paths: Allowed executable path prefixes.
    :param cdhashes: Allowed cdhash values.
    :param verbose: Enables additional runtime logging.
    :param allow_all: Forces all validations to pass regardless of path/cdhash.
    """
    ret_reg, self_reg, error_out_reg = registers_for_target(target)

    frame = thread.frames[0]
    frame_name = frame.GetFunctionName() or ""
    if frame_name not in VALIDATOR_METHODS:
        return
    if verbose:
        hit_pc = frame.GetPCAddress().GetLoadAddress(target)
        hit_module = frame.GetModule().GetFileSpec().GetFilename()
        print(f"Hit breakpoint: {frame_name} @ {hex(hit_pc)} ({hit_module})")

    validator = frame.reg[self_reg].value
    error_out_ptr = parse_pointer(frame.reg[error_out_reg].value)
    result = dump_validator(target, validator, preferred_paths=paths)
    should_allow = False
    allow_reason = ""

    if allow_all:
        should_allow = True
        allow_reason = f"--allow-all: {result['path']}"
    elif result["cdhash"] in cdhashes:
        should_allow = True
        allow_reason = f"cdhash {result['cdhash']}"
    else:
        for path in paths:
            if result["path"].startswith(path):
                should_allow = True
                allow_reason = f"path {result['path']}"
                break

    if should_allow:
        prepare_validator_for_bypass(target, validator, frame_name)
        patched_callsite = patch_validate_callsite_isvalid_store(target, frame, verbose=verbose)
        clear_pointer_value(target.GetProcess(), error_out_ptr)
        if patched_callsite:
            thread.StepOutOfFrame(frame)
            force_validator_success(target, validator)
        else:
            returned_early = force_return_success(target, thread, frame, ret_reg, verbose=verbose)
            if verbose and returned_early:
                print("Bypass strategy: ReturnFromFrame early-success")
            if not returned_early:
                thread.StepOutOfFrame(frame)
                current_frame = thread.frames[0]
                ret = current_frame.reg[ret_reg]
                if verbose:
                    print(f"Return register before patch: {ret_reg}={ret.value}")
                write_ok = write_register_value(target, thread, current_frame, ret_reg, "1")
                if verbose:
                    updated_ret = current_frame.reg[ret_reg]
                    print(f"Return register patch status: {write_ok}, after={updated_ret.value}")
            force_validator_success(target, validator)
        if verbose:
            print(f"Allowed due to {allow_reason}")
        return

    # Let real validation run for unmatched binaries.
    thread.StepOutOfFrame(frame)
    post_result = dump_validator(target, validator, preferred_paths=paths)
    if verbose and not post_result["is_valid"]:
        print("Invalid path not patched:")
        pprint(post_result)


def force_validator_success(target: lldb.SBTarget, validator: str) -> None:
    """
    Force key AMFIPathValidator state bits to a "validated success" shape.
    """
    process = target.GetProcess()
    try:
        validator_ptr = int(validator, 0)
    except (TypeError, ValueError):
        return

    if validator_ptr == 0:
        return

    edits = (
        (AMFI_VALIDATOR_VALIDATED_OFFSET, b"\x01"),  # _validated
        (AMFI_VALIDATOR_IS_SIGNED_OFFSET, b"\x01"),  # _isSigned
        (AMFI_VALIDATOR_IS_VALID_OFFSET, b"\x01"),  # _isValid
        (AMFI_VALIDATOR_ARE_ENTITLEMENTS_VALIDATED_OFFSET, b"\x01"),  # _areEntitlementsValidated
    )
    for offset, payload in edits:
        error = lldb.SBError()
        process.WriteMemory(validator_ptr + offset, payload, error)
    clear_pointer_value(process, validator_ptr + AMFI_VALIDATOR_ERROR_OBJECT_OFFSET)


def prepare_validator_for_bypass(target: lldb.SBTarget, validator: str, frame_name: str) -> None:
    """
    Pre-adjust validator flags before validateWithError executes.
    """
    process = target.GetProcess()
    try:
        validator_ptr = int(validator, 0)
    except (TypeError, ValueError):
        return

    if validator_ptr == 0:
        return

    # macOS path validator tracks restricted-entitlement mode in these bytes.
    if "AMFIPathValidator_macos" in frame_name:
        for offset in (
            AMFI_VALIDATOR_HAS_RESTRICTED_ENTITLEMENTS_OFFSET,
            AMFI_VALIDATOR_HAS_ONLY_SOFT_RESTRICTED_ENTITLEMENTS_OFFSET,
        ):  # _hasRestrictedEntitlements, _hasOnlySoftRestrictedEntitlements
            error = lldb.SBError()
            process.WriteMemory(validator_ptr + offset, b"\x00", error)


def dump_validator(
    target: lldb.SBTarget,
    validator: str,
    preferred_paths: Optional[Set[str]] = None,
) -> Dict[str, Union[str, bool]]:
    """
    Read path/cdhash/validity fields from an `AMFIPathValidator` object.

    :param target: LLDB target attached to `amfid`.
    :param validator: Objective-C object pointer for `AMFIPathValidator`.
    :return: A dictionary with keys `path`, `cdhash`, and `is_valid`.
    """
    global PATH_POINTER_CHAIN, CDHASH_POINTER_CHAIN, CDHASH_BY_PATH_CACHE

    process = target.GetProcess()
    preferred_paths = preferred_paths or set()

    def eval_u64(expr: str) -> int:
        value = target.EvaluateExpression(expr)
        if not value.GetError().Success():
            return 0
        return int(value.unsigned)

    def read_c_string(addr: int, max_len: int = 4096) -> Optional[str]:
        if addr == 0:
            return None
        error = lldb.SBError()
        try:
            value = process.ReadCStringFromMemory(addr, max_len, error)
        except Exception:
            return None
        if not error.Success() or not value:
            return None
        return value

    def read_memory(addr: int, size: int) -> bytes:
        if addr == 0 or size <= 0:
            return b""
        error = lldb.SBError()
        try:
            data = process.ReadMemory(addr, size, error)
        except Exception:
            return b""
        if not error.Success() or not data:
            return b""
        return data

    def read_u64(addr: int) -> int:
        raw = read_memory(addr, 8)
        if len(raw) != 8:
            return 0
        return struct.unpack("<Q", raw)[0]

    def cdhash_from_codesign(path: str) -> Optional[str]:
        cached = CDHASH_BY_PATH_CACHE.get(path)
        if cached:
            return cached
        try:
            result = subprocess.run(
                ["codesign", "-d", "-vvv", path],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None

        output = (result.stderr or "") + "\n" + (result.stdout or "")
        for line in output.splitlines():
            if line.startswith("CDHash="):
                value = line.split("=", 1)[1].strip().lower()
                if re.fullmatch(r"[0-9a-f]{40}", value):
                    CDHASH_BY_PATH_CACHE[path] = value
                    return value
        return None

    def pointer_plausible(value: int) -> bool:
        return 0x100000000 <= value <= 0x0000FFFFFFFFFFFF

    def looks_like_path(text: str) -> bool:
        return text.startswith("/") and len(text) < 4096 and " " not in text

    def path_score(text: str) -> int:
        if not looks_like_path(text):
            return -1000

        score = 0
        if text.startswith("/Users/"):
            score += 120
        if text.startswith("/private/"):
            score += 80
        if text.startswith("/Applications/") or text.startswith("/System/Applications/"):
            score += 70
        if "/.build/" in text:
            score += 50
        if ".app/" in text or text.endswith(".app"):
            score += 40
        if text.endswith(".dylib") or text.endswith(".framework"):
            score += 20
        if text.startswith("/com.apple."):
            score -= 300
        if text.count("/") < 2:
            score -= 150
        if len(text) < 12:
            score -= 80
        if any(text.startswith(prefix) for prefix in preferred_paths):
            score += 10000
        return score

    def follow_chain(root: int, chain: Tuple[int, ...]) -> int:
        current = root
        for offset in chain:
            if not pointer_plausible(current):
                return 0
            current = read_u64(current + offset)
            if not pointer_plausible(current):
                return 0
        return current

    def discover_path_chain(root: int, max_depth: int = 6) -> Tuple[Optional[str], Optional[Tuple[int, ...]]]:
        best_path: Optional[str] = None
        best_chain: Optional[Tuple[int, ...]] = None
        best_score = -1000
        queue = deque([(root, tuple(), 0)])
        seen: Set[int] = set()

        while queue:
            current, chain, depth = queue.popleft()
            if current in seen or not pointer_plausible(current):
                continue
            seen.add(current)

            direct = read_c_string(current, 4096)
            if direct:
                score = path_score(direct)
                if score > best_score:
                    best_score = score
                    best_path = direct
                    best_chain = chain

            if depth >= max_depth:
                continue

            blob = read_memory(current, 0x200)
            if not blob:
                continue

            for offset in range(0, len(blob) - 7, 8):
                next_ptr = struct.unpack_from("<Q", blob, offset)[0]
                if pointer_plausible(next_ptr):
                    queue.append((next_ptr, chain + (offset,), depth + 1))

        if best_score < 0:
            return None, None
        return best_path, best_chain

    def discover_data_chain(
        root: int,
        target_bytes: bytes,
        max_depth: int = 6,
    ) -> Optional[Tuple[int, ...]]:
        if not target_bytes:
            return None

        queue = deque([(root, tuple(), 0)])
        seen: Set[int] = set()
        target_len = len(target_bytes)

        while queue:
            current, chain, depth = queue.popleft()
            if current in seen or not pointer_plausible(current):
                continue
            seen.add(current)

            if read_memory(current, target_len) == target_bytes:
                return chain

            if depth >= max_depth:
                continue

            blob = read_memory(current, 0x200)
            if not blob:
                continue

            for offset in range(0, len(blob) - 7, 8):
                next_ptr = struct.unpack_from("<Q", blob, offset)[0]
                if pointer_plausible(next_ptr):
                    queue.append((next_ptr, chain + (offset,), depth + 1))

        return None

    is_valid = bool(
        eval_u64(
            f"*(unsigned char *)((uintptr_t){validator} + {AMFI_VALIDATOR_IS_VALID_OFFSET})"
        )
    )

    try:
        validator_ptr = int(validator, 0)
    except (TypeError, ValueError):
        validator_ptr = 0

    path: Optional[str] = None
    if validator_ptr and PATH_POINTER_CHAIN is not None:
        path_ptr = follow_chain(validator_ptr, PATH_POINTER_CHAIN)
        path = read_c_string(path_ptr, 4096)
        if path is not None and path_score(path) < 0:
            path = None

    if path is None and validator_ptr:
        discovered_path, discovered_chain = discover_path_chain(validator_ptr)
        path = discovered_path
        if discovered_chain is not None and discovered_path is not None:
            if not preferred_paths or any(discovered_path.startswith(prefix) for prefix in preferred_paths):
                PATH_POINTER_CHAIN = discovered_chain

    cdhash: Optional[str] = None
    if validator_ptr and CDHASH_POINTER_CHAIN is not None:
        cdhash_ptr = follow_chain(validator_ptr, CDHASH_POINTER_CHAIN)
        raw = read_memory(cdhash_ptr, 20)
        if len(raw) == 20:
            cdhash = raw.hex()

    # Fallbacks for older/newer layouts where ivar offsets may differ.
    if path is None:
        path_desc = target.EvaluateExpression(
            f"(NSURL*)[(id){validator} mainExecutable]"
        ).GetObjectDescription()
        if path_desc and path_desc.startswith("file://"):
            path = path_desc[len("file://"):]
        else:
            legacy_path_desc = target.EvaluateExpression(
                f"(NSURL*)[(id){validator} codePath]"
            ).GetObjectDescription()
            if legacy_path_desc and legacy_path_desc.startswith("file://"):
                path = legacy_path_desc[len("file://"):]

    if cdhash is None:
        legacy_cdhash = target.EvaluateExpression(
            f"(NSData*)[(id){validator} cdhashAsData]"
        )
        legacy_cdhash_desc = legacy_cdhash.GetObjectDescription()
        if legacy_cdhash_desc:
            cdhash = legacy_cdhash_desc[1:-1].replace(" ", "")
            if validator_ptr and CDHASH_POINTER_CHAIN is None:
                try:
                    cdhash_bytes = bytes.fromhex(cdhash)
                except ValueError:
                    cdhash_bytes = b""
                if cdhash_bytes:
                    discovered_cdhash_chain = discover_data_chain(validator_ptr, cdhash_bytes)
                    if discovered_cdhash_chain is not None:
                        CDHASH_POINTER_CHAIN = discovered_cdhash_chain

    if cdhash is None and path:
        cdhash = cdhash_from_codesign(path)
        if cdhash and validator_ptr and CDHASH_POINTER_CHAIN is None:
            try:
                cdhash_bytes = bytes.fromhex(cdhash)
            except ValueError:
                cdhash_bytes = b""
            if cdhash_bytes:
                discovered_cdhash_chain = discover_data_chain(validator_ptr, cdhash_bytes)
                if discovered_cdhash_chain is not None:
                    CDHASH_POINTER_CHAIN = discovered_cdhash_chain

    if path is None:
        path = ""
    if cdhash is None:
        cdhash = ""

    return {
        "path": path,
        "cdhash": cdhash,
        "is_valid": is_valid,
    }


def get_stopped_thread(process: lldb.SBProcess, reason: int) -> Optional[lldb.SBThread]:
    """
    Return the first thread stopped for the specified LLDB stop reason.

    :param process: Active LLDB process object.
    :param reason: LLDB stop reason constant (for example, breakpoint).
    :return: The matching thread if found, otherwise `None`.
    """
    for thread in process:
        if thread.GetStopReason() == reason:
            return thread
    return None


def print_verbose_list(header: str, values: Set[str]) -> None:
    """
    Print a sorted verbose list with a stable empty-state output.

    :param header: Human-readable list title.
    :param values: Items to print.
    """
    print(f"  {header}:")
    if values:
        for value in sorted(values):
            print(f"    - {value}")
    else:
        print("    - (none)")


def bypass_loop(
    process: lldb.SBProcess,
    target: lldb.SBTarget,
    paths: Optional[List[str]] = None,
    cdhashes: Optional[List[str]] = None,
    verbose: bool = False,
    allow_all: bool = False,
) -> None:
    """
    Main runtime loop that continues `amfid`, reloads config changes, and patches
    matching validation results at the breakpoint.

    :param process: Attached LLDB process for `amfid`.
    :param target: LLDB target associated with `process`.
    :param paths: Optional allowlisted executable path prefixes.
    :param cdhashes: Optional allowlisted cdhash values.
    :param verbose: Enables verbose runtime logging.
    :param allow_all: Forces all validations to pass regardless of config.
    """
    cli_paths = set(paths or [])
    cli_cdhashes = set(cdhashes or [])
    config = load_persistent_config()
    modified_time_state = config_modified_time_state()
    allowed_paths = set(config["paths"]) | cli_paths
    allowed_cdhashes = set(config["cdhashes"]) | cli_cdhashes

    if verbose:
        print("Running configuration:")
        print(f"  Allow all: {allow_all}")
        print_verbose_list("Paths", allowed_paths)
        print_verbose_list("CDHashes", allowed_cdhashes)

    while True:
        process.Continue()
        if process.state not in [
            lldb.eStateRunning,
            lldb.eStateStopped,
            lldb.eStateSuspended,
        ]:
            raise RuntimeError(
                f"Unexpected process state {process.state}"
            )

        current_modified_time_state = config_modified_time_state()
        if current_modified_time_state != modified_time_state:
            config = load_persistent_config()
            allowed_paths = set(config["paths"]) | cli_paths
            allowed_cdhashes = set(config["cdhashes"]) | cli_cdhashes
            modified_time_state = config_modified_time_state()
            if verbose:
                print("Reloaded configuration from ~/.amfidont")

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        if thread:
            validate_hook(
                target,
                thread,
                allowed_paths,
                allowed_cdhashes,
                verbose=verbose,
                allow_all=allow_all,
            )


def run_bypass(
    paths: Optional[List[str]] = None,
    cdhashes: Optional[List[str]] = None,
    verbose: bool = False,
    allow_all: bool = False,
) -> None:
    """
    Run the foreground bypass loop against `amfid`.

    :param paths: Optional path prefixes supplied by CLI.
    :param cdhashes: Optional cdhashes supplied by CLI.
    :param verbose: Enables informative runtime logging.
    :param allow_all: Forces all validations to pass regardless of path/cdhash.
    """
    global ACTIVE_DEBUGGER, ACTIVE_VERBOSE
    debugger = lldb.SBDebugger.Create()
    ACTIVE_DEBUGGER = debugger
    ACTIVE_VERBOSE = verbose
    debugger.SetAsync(False)
    target = debugger.CreateTarget("")
    process = target.AttachToProcessWithName(
        debugger.GetListener(), AMFID_PATH, False, lldb.SBError()
    )

    if not process:
        print("Failed to attach to process, should probably run as root")
        return

    if verbose:
        print(f"Attached to {AMFID_PATH}")

    symbolic_bp = target.BreakpointCreateByName(
        "-[AMFIPathValidator_macos validateWithError:]",
        "AppleMobileFileIntegrity",
    )
    breakpoints = [symbolic_bp]
    if symbolic_bp.GetNumLocations() > 0:
        location = symbolic_bp.GetLocationAtIndex(0)
        load_addr = location.GetAddress().GetLoadAddress(target)
        address_bp = target.BreakpointCreateByAddress(load_addr)
        target.BreakpointDelete(symbolic_bp.GetID())
        breakpoints = [address_bp]
    if verbose:
        total_locations = 0
        for bp in breakpoints:
            total_locations += bp.GetNumLocations()
        print(f"Installed validateWithError breakpoints ({total_locations} locations)")

    try:
        thread = threading.Thread(
            target=bypass_loop,
            args=(process, target, paths, cdhashes, verbose, allow_all),
        )
        thread.daemon = True
        thread.start()
        thread.join()
    except KeyboardInterrupt:
        if verbose:
            print("Stopping amfidont (detaching from amfid)...")
