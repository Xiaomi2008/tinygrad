# live view of a `python -m tinygrad.llm --serve` server: prefill (PP) and decode (TG) speed of the request in progress, and recent ones
# usage: python -m tinygrad.llm.monitor [status file]   (the server writes it: LLM_STATUS_FILE, default <tempdir>/tinyllm_status.json)
import json, os, pathlib, shutil, sys, tempfile, time

DEFAULT = pathlib.Path(os.environ.get("LLM_STATUS_FILE", str(pathlib.Path(tempfile.gettempdir()) / "tinyllm_status.json")))
B, DIM, G, Y, C, R, X = "\x1b[1m", "\x1b[2m", "\x1b[32m", "\x1b[33m", "\x1b[36m", "\x1b[31m", "\x1b[0m"

def bar(frac:float, width:int) -> str:
  n = max(0, min(width, round(frac * width)))
  return G + "█" * n + DIM + "░" * (width - n) + X

def alive(pid:int|None) -> bool:
  if not pid: return False
  try: os.kill(pid, 0)
  except ProcessLookupError: return False
  except PermissionError: return True
  return True

def ago(t:float) -> str:
  s = time.time() - t
  return f"{s:.0f}s ago" if s < 120 else f"{s/60:.0f}m ago" if s < 7200 else f"{s/3600:.0f}h ago"

def render(st:dict|None, path:pathlib.Path) -> list[str]:
  width = shutil.get_terminal_size((100, 30)).columns
  if st is None: return [f"{B}TinyLLM monitor{X}", "", f"{Y}waiting for {path}{X} (start the server: it writes this file)"]
  up = alive(st.get("pid"))
  head = f"{B}TinyLLM monitor{X}  {st.get('model', '?')}  {DIM}ctx {st.get('max_context', 0):,}  pid {st.get('pid')}{X}  "
  lines = [head + (f"{G}● running{X}" if up else f"{R}● server not running{X}"), DIM + "─" * min(width, 100) + X]
  now, phase = time.time(), st.get("phase")
  if up and phase == "prefill":
    todo, done = st["prompt"] - st["cached"], st["done"] - st["cached"]
    el = max(st["updated"] - st["start"], 1e-6)
    rate = done / el if done > 0 else 0
    eta = f"  ~{(todo - done) / rate:.0f}s left" if rate > 0 and done < todo else ""
    lines += [f"{B}{C}▶ PREFILL{X}  {done:,} / {todo:,} new tokens  {DIM}({st['cached']:,} cached){X}",
              f"  {bar(done / max(todo, 1), 40)} {100 * done / max(todo, 1):3.0f}%   {B}PP {rate:6.0f} tok/s{X}   {now - st['start']:5.1f}s{eta}"]
  elif up and phase == "decode":
    el = max(st["updated"] - st["decode_start"], 1e-6)
    rate = st["gen"] / el if st["gen"] > 1 else 0
    cap = f" / {st['max_tokens']:,}" if st.get("max_tokens") else ""
    lines += [f"{B}{G}▶ DECODE{X}   {st['gen']:,}{cap} tokens   {B}TG {rate:5.1f} tok/s{X}   {now - st['decode_start']:5.1f}s",
              f"  {DIM}prompt {st['prompt']:,} tokens ({st['cached']:,} cached) read at {st.get('prefill_tps', 0):.0f} tok/s, "
              f"context {st['prompt'] + st['gen']:,} / {st.get('max_context', 0):,}{X}"]
  else:
    last = st.get("last")
    lines += [f"{DIM}● idle{X}" + (f"{DIM}, last request finished {ago(last['end'])}{X}" if last else ""), ""]
  hist = st.get("history") or []
  lines += ["", f"{B}recent requests{X}",
            f"{DIM}{'time':>8}  {'prompt':>7}  {'cached':>7}  {'new':>7}  {'PP tok/s':>8}  {'out':>6}  {'TG tok/s':>8}  {'total':>7}{X}"]
  for h in hist[-max(3, shutil.get_terminal_size((100, 30)).lines - 14):]:
    flag = f" {R}interrupted{X}" if h.get("interrupted") else ""
    lines.append(f"{time.strftime('%H:%M:%S', time.localtime(h['end'])):>8}  {h['prompt']:7,}  {h['cached']:7,}  {h['prompt'] - h['cached']:7,}  "
                 f"{h['prefill_tps']:8.0f}  {h['gen']:6,}  {h['gen_tps']:8.1f}  {h['total_s']:6.1f}s{flag}")
  # averages over requests big enough to measure: short prompts and replies are dominated by fixed overhead
  pp = [h["prefill_tps"] for h in hist if h["prompt"] - h["cached"] >= 512]
  tg = [h["gen_tps"] for h in hist if h["gen"] >= 32]
  if pp or tg:
    lines += ["", f"{B}average{X}  PP {sum(pp)/len(pp):.0f} tok/s {DIM}(prompts ≥512 new tokens, n={len(pp)}){X}" if pp else "",
              f"         TG {sum(tg)/len(tg):.1f} tok/s {DIM}(replies ≥32 tokens, n={len(tg)}){X}" if tg else ""]
  return lines

def main():
  path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
  sys.stdout.write("\x1b[?1049h\x1b[?25l")  # alternate screen, hide cursor
  try:
    while True:
      try: st = json.loads(path.read_text())
      except (FileNotFoundError, json.JSONDecodeError): st = None
      sys.stdout.write("\x1b[H" + "\n".join(line + "\x1b[K" for line in render(st, path)) + "\x1b[J")
      sys.stdout.flush()
      time.sleep(0.25)
  except KeyboardInterrupt: pass
  finally: sys.stdout.write("\x1b[?25h\x1b[?1049l")

if __name__ == "__main__": main()
