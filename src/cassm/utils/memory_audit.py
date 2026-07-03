# memory_audit.py (post-line attribution + per-line deltas)
import sys, inspect
from collections import defaultdict, deque
import torch
from torch.utils._python_dispatch import TorchDispatchMode

# -------- storage helpers --------
def storage_key(t: torch.Tensor):
    try:
        return (t.device.type, int(t.untyped_storage().data_ptr()))
    except Exception:
        return (t.device.type, id(t))

def storage_bytes(t: torch.Tensor) -> int:
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:
        return int(t.element_size() * t.numel())

def tensor_bytes(t: torch.Tensor) -> int:
    # logical size (ignores storage sharing)
    return int(t.element_size() * t.numel())

def walk_tensors(obj, fn):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    tot = 0
    if isinstance(obj, (list, tuple)):
        for e in obj: tot += walk_tensors(e, fn)
    elif isinstance(obj, dict):
        for e in obj.values(): tot += walk_tensors(e, fn)
    return tot

# -------- combined tracer --------
class _MemDispatch(TorchDispatchMode):
    """
    Tracks op outputs. Calls back to CombinedMemoryAudit with:
      on_op(added_unique_bytes, added_sum_bytes, lineno, op_name, shapes, running_unique_bytes)
    """
    def __init__(self, lineno_getter, on_op_cb):
        super().__init__()
        self._lineno = lineno_getter
        self._cb = on_op_cb
        self.seen_ptrs = set()
        self.running_unique = 0  # sum of new storages we've seen

    def _accumulate(self, out):
        # bytes of NEW storages
        added_unique = 0
        def add_unique(t: torch.Tensor):
            nonlocal added_unique
            k = storage_key(t)
            if k not in self.seen_ptrs:
                self.seen_ptrs.add(k)
                added_unique += storage_bytes(t)
            return 0

        added_sum = 0
        def add_sum(t: torch.Tensor):
            nonlocal added_sum
            added_sum += tensor_bytes(t)  # count logical size always
            return 0

        walk_tensors(out, add_unique)
        walk_tensors(out, add_sum)

        self.running_unique += added_unique

        shapes = []
        def collect_shape(t: torch.Tensor):
            shapes.append(tuple(t.shape))
            return 0
        walk_tensors(out, collect_shape)

        if added_unique or added_sum:
            ln = self._lineno()
            self._cb(added_unique, added_sum, ln, shapes)
        return added_unique, added_sum

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None: kwargs = {}
        out = func(*args, **kwargs)
        self._accumulate(out)
        return out

class CombinedMemoryAudit:
    """
    Post-line memory attribution that combines:
      - per-op deltas (unique + raw sum) grouped by source line,
      - running unique-bytes snapshot *after* each line executes,
      - optional locals-bytes snapshot (after-line).
    Use:
        audit = CombinedMemoryAudit(model.filter)  # or model.forward
        out = audit.run(model, data, return_type="for_forward")
        audit.report_by_line(top_ops=3)    # main table
        audit.report_top_spikes(top=20)    # biggest op bursts with line tags
    """
    def __init__(self, target_func):
        self.target_func = target_func
        self._code = target_func.__code__
        self._src = inspect.getsource(target_func).splitlines()
        self._first = self._code.co_firstlineno

        # state
        self._current_lineno = None
        self._pending_added_unique = 0
        self._pending_added_sum = 0
        self._running_unique = 0

        # records per executed line (in order):
        # dicts with: lineno, code, added_unique, added_sum, running_unique_after, locals_bytes_after
        self.line_records = []

        # per-line op details
        self._line_added_unique = defaultdict(int)
        self._line_added_sum = defaultdict(int)

        # also keep a spike log (ordered)
        self.spikes = []  # (added_unique, added_sum, lineno, shapes, running_unique_after)

    # ---- callbacks from MemDispatch ----
    def _on_op(self, added_unique, added_sum, lineno, shapes):
        if self._current_lineno is None:
            # before first line event - attribute to function entry (rare)
            return
        self._pending_added_unique += added_unique
        self._pending_added_sum += added_sum
        self._line_added_unique[self._current_lineno] += added_unique
        self._line_added_sum[self._current_lineno] += added_sum
        # running unique is maintained in dispatch; we read it on flush

    # ---- tracer that flushes AFTER each line has executed ----
    def _tracer(self, frame, event, arg):
        if frame.f_code is not self._code:
            return self._tracer

        if event == 'line':
            # We are ABOUT to run a new line => previous line just finished.
            # Flush the previous line’s totals using the current frame state.
            self._flush_line(frame)
            # Set new current line
            self._current_lineno = frame.f_lineno
            return self._tracer

        if event == 'return':
            # Function is returning; flush the last line too.
            self._flush_line(frame, is_return=True)
            return self._tracer

        return self._tracer

    def _flush_line(self, frame, is_return=False):
        if self._current_lineno is None:
            # this is the very first line; nothing to flush yet
            return
        # running unique so far (from dispatch)
        self._running_unique = self.mem_dispatch.running_unique
        # locals snapshot AFTER the previous line executed (we're at next line/return)
        locals_bytes_after = self._locals_bytes(frame.f_locals)
        # prepare record
        idx = self._current_lineno - self._first
        code_line = self._src[idx].rstrip() if 0 <= idx < len(self._src) else ""
        rec = dict(
            lineno=self._current_lineno,
            code=code_line,
            added_unique=self._pending_added_unique,
            added_sum=self._pending_added_sum,
            running_unique_after=self._running_unique,
            locals_bytes_after=locals_bytes_after
        )
        if self._pending_added_unique or self._pending_added_sum:
            # record a spike row for convenience
            self.spikes.append((
                self._pending_added_unique,
                self._pending_added_sum,
                self._current_lineno,
                [],  # shapes are summarized per-op; not stored here
                self._running_unique
            ))
        self.line_records.append(rec)
        # reset pending for next line
        self._pending_added_unique = 0
        self._pending_added_sum = 0

    @staticmethod
    def _locals_bytes(locals_dict):
        # sum logical sizes of tensors present in locals (after line)
        seen = set()
        total = 0
        def add(t):
            nonlocal total
            k = storage_key(t)
            if k in seen: return
            seen.add(k)
            total += tensor_bytes(t)
        for v in locals_dict.values():
            if isinstance(v, torch.Tensor):
                add(v)
            elif isinstance(v, (list, tuple)):
                for e in v:
                    if isinstance(e, torch.Tensor): add(e)
            elif isinstance(v, dict):
                for e in v.values():
                    if isinstance(e, torch.Tensor): add(e)
        return total

    # ---- public run ----
    def run(self, obj, *args, **kwargs):
        # reset state
        self.line_records.clear()
        self.spikes.clear()
        self._line_added_unique.clear()
        self._line_added_sum.clear()
        self._current_lineno = None
        self._pending_added_unique = 0
        self._pending_added_sum = 0
        self._running_unique = 0

        # set up dispatch with callback
        self.mem_dispatch = _MemDispatch(lambda: self._current_lineno, self._on_op)

        old = sys.gettrace()
        sys.settrace(self._tracer)
        try:
            with self.mem_dispatch:
                return self.target_func.__get__(obj, obj.__class__)(*args, **kwargs)
        finally:
            sys.settrace(old)

    # ---- reports ----
    def report_by_line(self, to_mb=True, top_ops=0):
        unit = 1e6 if to_mb else 1.0
        print("\n=== Memory by line (post-line) ===")
        print(" line |  +uniq  |   +sum  | running | locals | code")
        for r in self.line_records:
            au = r['added_unique']/unit
            am = r['added_sum']/unit
            run = r['running_unique_after']/unit
            loc = r['locals_bytes_after']/unit
            print(f"{r['lineno']:5d} | {au:7.1f} | {am:7.1f} | {run:7.1f} | {loc:6.1f} | {r['code']}")
        # optional per-line totals
        if top_ops:
            print("\n(Per-line totals; top lines by +sum)")
            lines_sorted = sorted(self._line_added_sum.items(), key=lambda kv: kv[1], reverse=True)[:top_ops]
            for ln, s in lines_sorted:
                print(f"line {ln}: +sum {s/unit:.1f}{'MB' if to_mb else 'B'}  (+uniq {self._line_added_unique[ln]/unit:.1f})")

    def report_top_spikes(self, top=20, to_mb=True):
        unit = 1e6 if to_mb else 1.0
        # derive spikes from per-line records (sorted by added_sum)
        recs = sorted(
            (r for r in self.line_records if r['added_unique'] or r['added_sum']),
            key=lambda r: r['added_sum'],
            reverse=True
        )[:top]
        print("\n=== Top line spikes ===")
        for r in recs:
            print(f"line {r['lineno']:5d} | +uniq {r['added_unique']/unit:7.1f} | +sum {r['added_sum']/unit:7.1f} "
                  f"| running {r['running_unique_after']/unit:7.1f} | {r['code']}")
