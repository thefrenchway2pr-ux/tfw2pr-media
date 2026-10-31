"""TheFrenchWay2PR reel maker: reel.json -> 1080x1920 MP4 (voice, captions, music, sound effects) + cover JPEG.

Usage:
  python3 tfw_reel.py reel.json OUT_DIR --tts TTS_DIR --logo logo.png [--stills 0.5 3 8]
With "voice": "none" (the owner's choice for now) no voice is made: scenes last as long as their text takes to read,
the music is louder, "say" is ignored and --tts is not needed.
TTS_DIR holds kokoro-v1.0.int8.onnx and voices-v1.0.bin
(github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/<name>); pip install kokoro-onnx soundfile scipy.
Needs ffmpeg and the Poppins font (~/.fonts). Output: OUT_DIR/<asset_name>.mp4 and <asset_name>_cover.jpg.

reel.json:
{ "asset_name": "2026-11-04_Reel_Je-Suis-30-Ans",
  "style": "mistakes",                    # mistakes | technique | serious | funny (sets music + pacing)
  "voice": "af_heart",                    # optional Kokoro English voice
  "scenes": [
    {"type": "hook",  "text": "STOP saying\n[r]« Je suis 30 ans »[/r]", "say": "Stop saying je suis trente ans!"},
    {"type": "wrong", "text": "Je suis 30 ans.", "say": "Je suis trente ans.", "lang": "fr"},
    {"type": "right", "text": "J'[b]ai[/b] 30 ans.", "sub": "I'm 30 years old.", "say": "J'ai trente ans.", "lang": "fr"},
    {"type": "tip",   "label": "Why?", "text": "In French, you [b]have[/b] an age.", "say": "In French, you have your age."},
    {"type": "chat",  "who": "left", "name": "Priya", "text": "Tu as quel âge ?", "say": "Tu as quel âge ?", "lang": "fr"},
    {"type": "cta",   "text": "Save this &\nfollow for more", "say": "Save this, and follow for daily French."}
  ] }
Every scene has "say" (spoken; "lang" en or fr, default en) and "text" (on screen; [b]..[/b] blue, [r]..[/r] red, \\n new line).
"say" may also be a list of {"say","lang"} parts (e.g. English then French). Keep reels 20-40 s, the hook under 2 s.
"""
import argparse, json, math, re, subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.signal import butter, sosfilt, resample_poly, fftconvolve

FPS, W, H, SR = 30, 1080, 1920, 48000
NAVY, NAVY2, BLUE, LBLUE, RED, LRED = (11, 28, 74), (20, 46, 110), (0, 85, 164), (122, 162, 255), (225, 37, 27), (255, 90, 78)
WHITE, CREAM, INK, GREEN = (255, 255, 255), (250, 246, 236), (20, 26, 48), (22, 150, 90)

_FONTS = {}
def font(size, weight='Bold'):
    """Poppins in the given weight (Regular, Medium, SemiBold, Bold, ExtraBold), from ~/.fonts or fontconfig"""
    key = (size, weight)
    if key not in _FONTS:
        num = {'Regular': 400, 'Medium': 500, 'SemiBold': 600, 'Bold': 700, 'ExtraBold': 800}[weight]
        cands = [Path.home() / f'.fonts/Poppins-{weight}.ttf', Path.home() / f'.fonts/Poppins-{num}-latin.ttf',
                 Path(f'/usr/share/fonts/truetype/poppins/Poppins-{weight}.ttf')]
        path = next((str(c) for c in cands if c.exists()), None)
        if not path:
            path = subprocess.run(['fc-match', '-f', '%{file}', f'Poppins:weight={ {400: 80, 500: 100, 600: 180, 700: 200, 800: 205}[num] }'],
                                  capture_output=True, text=True).stdout or None
        _FONTS[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    return _FONTS[key]

def ease(x): x = min(1, max(0, x)); return 1 - (1 - x) ** 3
def back(x):  # overshoot pop
    x = min(1, max(0, x)); c = 1.9; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2

def runs(text):
    """'a [b]b[/b] c' -> [(chunk, color)] per line"""
    lines = []
    for ln in text.split('\n'):
        out = []
        for part in re.split(r'(\[b\].*?\[/b\]|\[r\].*?\[/r\])', ln):
            if not part: continue
            if part.startswith('[b]'): out.append((part[3:-4], 'b'))
            elif part.startswith('[r]'): out.append((part[3:-4], 'r'))
            else: out.append((part, None))
        lines.append(out)
    return lines

def plain(text): return re.sub(r'\[/?[br]\]', '', text)

def draw_rich(img, text, cx, cy, size, base=WHITE, b=LBLUE, r=LRED, weight='ExtraBold', maxw=940, alpha=255, chars=None):
    """centered multi-line rich text; shrinks to fit maxw; chars limits how many characters show (typewriter)"""
    d = ImageDraw.Draw(img)
    while True:
        f = font(size, weight); lines = runs(text)
        widths = [sum(d.textlength(c, font=f) for c, _ in ln) for ln in lines]
        if max(widths + [0]) <= maxw or size < 30: break
        size -= 4
    lh = int(size * 1.18); y = cy - lh * (len(lines) - 1) / 2; left = chars if chars is not None else 10 ** 6
    for ln, wd in zip(lines, widths):
        x = cx - wd / 2
        for c, k in ln:
            if left <= 0: break
            c = c[:left]; left -= len(c)
            col = {'b': b, 'r': r}.get(k, base)
            d.text((x, y), c, font=f, fill=col + (alpha,) if img.mode == 'RGBA' else col, anchor='lm')
            x += d.textlength(c, font=f)
        y += lh
    return size

# ---------------- voice ----------------
def parts(sc):
    s = sc.get('say', '')
    if isinstance(s, list): return [(p['say'], p.get('lang', 'en')) for p in s]
    return [(s, sc.get('lang', 'en'))] if s else []

def make_voice(spec, tts_dir, work):
    if spec.get('voice') == 'none': return [[] for _ in spec['scenes']]
    from kokoro_onnx import Kokoro
    k = Kokoro(str(Path(tts_dir) / 'kokoro-v1.0.int8.onnx'), str(Path(tts_dir) / 'voices-v1.0.bin'))
    en_voice = spec.get('voice', 'af_heart'); out = []
    for i, sc in enumerate(spec['scenes']):
        clips = []
        for j, (say, lang) in enumerate(parts(sc)):
            if lang == 'fr': a, sr = k.create(say, voice='ff_siwis', speed=0.95, lang='fr-fr')
            else: a, sr = k.create(say, voice=en_voice, speed=1.12, lang='en-us')
            a = resample_poly(a, SR, sr); nz = np.where(np.abs(a) > 0.01)[0]
            if len(nz): a = a[max(0, nz[0] - 800):nz[-1] + 1600]  # trim silence
            clips.append({'audio': a / (np.abs(a).max() + 1e-9) * 0.85, 'text': say, 'lang': lang})
        out.append(clips)
    return out

# ---------------- timeline ----------------
def plan(spec, voice):
    pace = {'funny': 0.18, 'mistakes': 0.22, 'technique': 0.28, 'serious': 0.32}.get(spec.get('style'), 0.25)
    t, scenes, vo, words = 0.15, [], [], []
    for sc, clips in zip(spec['scenes'], voice):
        st = t; v = st + (0.05 if sc['type'] == 'hook' else 0.2)
        for c in clips:
            d = len(c['audio']) / SR; vo.append((v, c['audio']))
            ws = c['text'].split(); wt = [len(w) + 2 + (4 if w[-1] in '.,?!:;' else 0) for w in ws]; s0 = v + 0.05
            for w, k in zip(ws, wt):
                dd = (d - 0.12) * k / sum(wt); words.append((len(scenes), w, s0, s0 + dd)); s0 += dd
            v += d + 0.1
        if not clips:  # no voice: give the viewer time to read the on-screen text
            n = len(plain(sc.get('text', '')).split()) + len((sc.get('sub') or '').split()) + len((sc.get('label') or '').split())
            v = st + {'hook': 1.0, 'cta': 1.6, 'wrong': 1.5, 'right': 1.4}.get(sc['type'], 0.9) + 0.3 * n
            if sc['type'] == 'tip': v += 0.4 * len(plain(sc.get('text', ''))) / 30
        hold = {'cta': 1.4, 'right': 0.6, 'wrong': 0.45, 'hook': 0.15}.get(sc['type'], pace)
        end = max(v - 0.1 + hold, st + {'hook': 1.2}.get(sc['type'], 1.0))
        scenes.append({**sc, 'start': 0.0 if not scenes else st, 'end': end}); t = end
    return scenes, vo, words, t

# ---------------- audio ----------------
def soundtrack(spec, scenes, vo, total, path):
    N = int((total + 0.3) * SR); rng = np.random.default_rng(7)
    def add(buf, st, sig):
        i = int(st * SR)
        if 0 <= i < len(buf): j = min(len(buf), i + len(sig)); buf[i:j] += sig[:j - i]
    lp = lambda x, f: sosfilt(butter(2, f, 'low', fs=SR, output='sos'), x)
    hp = lambda x, f: sosfilt(butter(2, f, 'high', fs=SR, output='sos'), x)
    midi = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    style = spec.get('style', 'mistakes')
    bpm = {'funny': 112, 'mistakes': 96, 'technique': 88, 'serious': 78}.get(style, 92); beat = 60 / bpm; bar = 4 * beat
    prog = {'serious': [[45, 52, 57, 60, 64], [41, 48, 53, 57, 60], [43, 50, 55, 59, 62], [40, 47, 52, 55, 59]],
            'funny': [[48, 55, 60, 64, 67], [53, 57, 60, 65, 69], [55, 59, 62, 67, 71], [48, 55, 60, 64, 67]]}.get(
        style, [[48, 55, 59, 62, 64], [45, 52, 55, 59, 62], [41, 48, 52, 55, 60], [43, 50, 53, 57, 62]])
    keys, low = np.zeros(N), np.zeros(N)
    for b in range(int(total / bar) + 2):
        ch = prog[b % 4]; st = b * bar
        hits = [(0, ch[1:]), (1.5, ch[2:]), (2.5, ch[1:4])] if style != 'funny' else [(0, ch[1:]), (1, ch[2:4]), (2, ch[1:]), (3, ch[3:])]
        for bt, notes in hits:
            n = int(1.4 * SR); tt = np.arange(n) / SR
            for m in notes:
                f = midi(m + 12); s = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * (4 if style == 'funny' else 2.8))
                add(keys, st + bt * beat + rng.uniform(0, 0.015), s * 0.03)
        n = int(bar * SR); tt = np.arange(n) / SR
        add(low, st, np.sin(2 * np.pi * midi(ch[0] - 12) * tt) * np.minimum(1, tt * 20) * np.exp(-tt * 0.8) * 0.16)
        if st < scenes[0]['end']: continue  # drums come in after the hook
        for k in range(4):
            n = int(0.35 * SR); tt = np.arange(n) / SR
            if k in (0, 2): add(low, st + k * beat, np.sin(2 * np.pi * (50 + 70 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 10) * 0.4)
            if k in (1, 3):
                m = int(0.2 * SR); add(low, st + k * beat, lp(hp(rng.standard_normal(m), 1200), 6000) * np.exp(-np.arange(m) / SR * 20) * 0.05)
            for h in (0, 0.5):
                m = int(0.04 * SR); add(low, st + (k + h) * beat + 0.01, hp(rng.standard_normal(m), 8000) * np.exp(-np.arange(m) / SR * 90) * 0.02)
    ir = rng.standard_normal(int(1.4 * SR)) * np.exp(-np.arange(int(1.4 * SR)) / SR * 3.5); ir /= np.abs(ir).sum() / 5
    music = keys + low + fftconvolve(keys, ir)[:N] * 0.35; music = lp(music, 7000); music /= np.abs(music).max() + 1e-9
    sfx = np.zeros(N)
    n = int(0.5 * SR); tt = np.arange(n) / SR  # hook impact
    add(sfx, 0.0, (np.sin(2 * np.pi * (45 + 90 * np.exp(-tt * 18)) * tt) * np.exp(-tt * 6) * 0.6 + lp(rng.standard_normal(n), 2500) * np.exp(-tt * 14) * 0.15))
    for sc in scenes[1:]:
        s = sc['start']
        if sc['type'] == 'wrong':
            n = int(0.35 * SR); tt = np.arange(n) / SR; add(sfx, s + 0.15, np.sign(np.sin(2 * np.pi * 140 * tt)) * np.exp(-tt * 9) * 0.07)
        elif sc['type'] == 'right':
            for i, f in enumerate((880, 1320)):
                n = int(0.25 * SR); tt = np.arange(n) / SR; add(sfx, s + 0.12 + i * 0.09, np.sin(2 * np.pi * f * tt) * np.exp(-tt * 14) * 0.12)
        elif sc['type'] == 'chat':
            n = int(0.12 * SR); tt = np.arange(n) / SR; add(sfx, s + 0.05, np.sin(2 * np.pi * (500 + 900 * tt / 0.12) * tt) * np.exp(-tt * 35) * 0.12)
        else:
            n = int(0.5 * SR); tt = np.arange(n) / SR; add(sfx, s - 0.25, lp(hp(rng.standard_normal(n), 300), 3000) * np.sin(np.pi * tt / 0.5) ** 2 * 0.05)
    voice, act = np.zeros(N), np.zeros(N)
    for st, a in vo:
        add(voice, st, hp(a, 80)); act[int((st - 0.1) * SR):int((st + len(a) / SR + 0.1) * SR)] = 1
    k = int(0.2 * SR); act = np.convolve(act, np.ones(k) / k, 'same')
    mix = music * (0.28 if vo else 0.6) * (1 - 0.5 * act) + sfx + voice
    f = int(0.8 * SR); mix[-f:] *= np.linspace(1, 0, f)
    sf.write(path, np.stack([mix, mix], 1) / (np.abs(mix).max() + 1e-9) * 0.95, SR)

# ---------------- pictures ----------------
class Painter:
    def __init__(self, scenes, words, total, logo_path, maple):
        self.scenes, self.words, self.total = scenes, words, total
        yy = np.linspace(0, 1, H)[:, None, None]; xx = np.linspace(0, 1, W)[None, :, None]
        g = np.array(NAVY2) * (1 - yy) * (1 - xx * 0.4) + np.array((7, 20, 54)) * yy + np.array(NAVY) * (1 - yy) * xx * 0.4
        self.bg = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).convert('RGBA')
        if maple:
            lf = Image.open(maple).convert('RGBA').resize((900, 900)); a = lf.getchannel('A').point(lambda v: int(v * 0.10))
            lf.putalpha(a); self.bg.paste(lf, (380, 160), lf)
        gl = Image.new('RGBA', (1200, 1200), (0, 0, 0, 0)); ImageDraw.Draw(gl).ellipse([250, 250, 950, 950], fill=(225, 37, 27, 70))
        self.glow = gl.filter(ImageFilter.GaussianBlur(150))
        logo = Image.open(logo_path).convert('RGBA'); m = Image.new('L', logo.size, 0)
        ImageDraw.Draw(m).ellipse([0, 0, logo.size[0] - 1, logo.size[1] - 1], fill=255); logo.putalpha(m); self.logo = logo
        d = ImageDraw.Draw(self.bg)
        for i, c in enumerate([BLUE, WHITE, RED]): d.rectangle([i * W // 3, H - 14, (i + 1) * W // 3, H], fill=c)
        lg = self.logo.resize((120, 120), Image.LANCZOS); self.bg.paste(lg, (60, 230), lg)
        d.text((196, 290), '@thefrenchway2pr', font=font(30, 'SemiBold'), fill=(210, 222, 255), anchor='lm')

    def caption(self, img, i, T):
        ws = [w for w in self.words if w[0] == i]
        if not ws: return
        chunks, cur = [], []
        for w in ws:
            cur.append(w)
            if len(cur) == 3 or w[1][-1] in '.,?!:;': chunks.append(cur); cur = []
        if cur: chunks.append(cur)
        for j, ch in enumerate(chunks):
            nxt = chunks[j + 1][0][2] if j + 1 < len(chunks) else ch[-1][3] + 0.35
            if ch[0][2] <= T < nxt:
                d = ImageDraw.Draw(img); f = font(54, 'ExtraBold'); txt = ' '.join(w[1] for w in ch)
                cur_w = [w for w in ch if w[2] <= T]
                tw = d.textlength(txt, font=f) + 60; s = 0.92 + 0.08 * ease((T - ch[0][2]) / 0.1)
                d.rounded_rectangle([(W - tw * s) / 2, 1500 - 50, (W + tw * s) / 2, 1500 + 50], 24, fill=(0, 0, 0, 150))
                x = (W - d.textlength(txt, font=f)) / 2
                for w in ch:
                    on = w is (cur_w[-1] if cur_w else None)
                    d.text((x, 1500), w[1], font=f, fill=(255, 214, 64) if on else WHITE, anchor='lm'); x += d.textlength(w[1] + ' ', font=f)
                return

    def card(self, w, h, fill=CREAM):
        c = Image.new('RGBA', (w + 60, h + 60), (0, 0, 0, 0)); d = ImageDraw.Draw(c)
        d.rounded_rectangle([30, 44, w + 30, h + 44], 36, fill=(0, 0, 0, 110)); c = c.filter(ImageFilter.GaussianBlur(14))
        ImageDraw.Draw(c).rounded_rectangle([30, 30, w + 30, h + 30], 36, fill=fill); return c

    def frame(self, T):
        for i, sc in enumerate(self.scenes):
            if sc['start'] <= T < sc['end'] or i == len(self.scenes) - 1: break
        t = T - sc['start']; img = self.bg.copy(); typ = sc['type']
        gx, gy = W * (0.5 + 0.35 * math.sin(T * 0.6)), H * (0.42 + 0.12 * math.cos(T * 0.45))  # slow moving glow
        img.alpha_composite(self.glow, (int(gx - 600), int(gy - 600)))
        d = ImageDraw.Draw(img)
        if typ == 'hook':
            s = back(t / 0.35); shake = int(10 * math.sin(t * 60) * max(0, 1 - t / 0.3))
            lay = Image.new('RGBA', (W, 900), (0, 0, 0, 0)); draw_rich(lay, sc['text'], W / 2, 450, 124, maxw=980)
            if sc.get('badge'):
                ld = ImageDraw.Draw(lay); f = font(46, 'ExtraBold'); bw = ld.textlength(sc['badge'].upper(), font=f) + 70
                ld.rounded_rectangle([(W - bw) / 2, 120, (W + bw) / 2, 210], 45, fill=RED + (255,))
                ld.text((W / 2, 165), sc['badge'].upper(), font=f, fill=WHITE, anchor='mm')
            lay = lay.resize((max(1, int(W * s)), max(1, int(900 * s))), Image.LANCZOS)
            img.paste(lay, (int((W - lay.size[0]) / 2) + shake, int(880 - lay.size[1] / 2)), lay)
            if t < 0.12: img.alpha_composite(Image.new('RGBA', (W, H), (255, 255, 255, int(160 * (1 - t / 0.12)))))
        elif typ in ('wrong', 'right'):
            a = ease(t / 0.3); bad = typ == 'wrong'
            c = self.card(900, 520); cd = ImageDraw.Draw(c)
            col = RED if bad else GREEN
            cd.ellipse([80, 80, 200, 200], fill=col); cd.text((140, 138), '✗' if bad else '✓', font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 76) if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf').exists() else font(70), fill=WHITE, anchor='mm')
            cd.text((230, 140), 'Not natural' if bad else 'Say this', font=font(44, 'Bold'), fill=col, anchor='lm')
            sz = draw_rich(c, sc['text'], 480, 330, 84, base=INK, b=BLUE, r=RED, maxw=800)
            if bad:  # strike-through drawn on
                p = ease((t - 0.45) / 0.35)
                if p > 0:
                    tw = min(800, cd.textlength(plain(sc['text']).split('\n')[0], font=font(sz, 'ExtraBold')))
                    cd.line([(480 - tw / 2, 330), (480 - tw / 2 + tw * p, 330)], fill=RED, width=12)
            if sc.get('sub'): cd.text((480, 470), sc['sub'], font=font(40, 'Medium'), fill=(80, 88, 120), anchor='mm')
            y = int(560 + 80 * (1 - a)); x = int((W - c.size[0]) / 2 + (0 if not bad else 18 * math.sin(t * 40) * max(0, 1 - (t - 0.5) / 0.3) * (t > 0.5)))
            c.putalpha(c.getchannel('A').point(lambda v: int(v * a))); img.paste(c, (x, y), c)
        elif typ == 'chat':
            left = sc.get('who', 'left') == 'left'; a = back(t / 0.3)
            prev = [s for s in self.scenes[:i] if s['type'] == 'chat'][-2:]  # keep the previous bubbles on screen
            y0 = 520
            for k, s2 in enumerate(prev + [sc]):
                l2 = s2.get('who', 'left') == 'left'; cur = s2 is sc; sa = a if cur else 1
                f = font(56, 'Bold'); txt = plain(s2['text']); tw = min(760, d.textlength(txt, font=f) + 80)
                bub = Image.new('RGBA', (int(tw) + 20, 170), (0, 0, 0, 0)); bd = ImageDraw.Draw(bub)
                bd.rounded_rectangle([10, 10, tw + 10, 160], 50, fill=(WHITE if l2 else BLUE) + (255,))
                draw_rich(bub, s2['text'], (tw + 20) / 2, 85, 56, base=INK if l2 else WHITE, b=BLUE if l2 else (255, 214, 64), r=RED, weight='Bold', maxw=tw - 60)
                bub = bub.resize((max(1, int(bub.size[0] * sa)), max(1, int(bub.size[1] * sa))), Image.LANCZOS)
                av = (90 if l2 else W - 90); ay = y0 + k * 260 + 80
                d.ellipse([av - 55, ay - 55, av + 55, ay + 55], fill=RED if l2 else (255, 196, 64))
                d.text((av, ay), (s2.get('name') or ('A' if l2 else 'B'))[0].upper(), font=font(56, 'ExtraBold'), fill=WHITE, anchor='mm')
                d.text((av, ay + 80), s2.get('name', ''), font=font(28, 'SemiBold'), fill=(210, 222, 255), anchor='mm')
                bx = 170 if l2 else W - 170 - bub.size[0]; img.paste(bub, (int(bx), int(ay - bub.size[1] / 2)), bub)
        elif typ == 'cta':
            a = ease(t / 0.5); sz = int(420 * (0.85 + 0.15 * a)); lg = self.logo.resize((sz, sz), Image.LANCZOS)
            img.paste(lg, (int((W - sz) / 2), 470), lg)
            draw_rich(img, sc['text'], W / 2, 1110, 84, maxw=940)
            fa = ease((t - 0.4) / 0.3)
            if fa > 0:
                tw = 520; d.rounded_rectangle([(W - tw) / 2, 1240, (W + tw) / 2, 1350], 55, fill=RED)
                d.text((W / 2, 1295), 'Follow @thefrenchway2pr', font=font(38, 'Bold'), fill=WHITE, anchor='mm')
        else:  # tip / text
            a = ease(t / 0.3)
            if sc.get('label'):
                f = font(40, 'Bold'); lw = d.textlength(sc['label'].upper(), font=f) + 60
                d.rounded_rectangle([(W - lw) / 2, 560 - 40 * a, (W + lw) / 2, 640 - 40 * a], 40, fill=RED)
                d.text((W / 2, 600 - 40 * a), sc['label'].upper(), font=f, fill=WHITE, anchor='mm')
            n = len(plain(sc['text'])); chars = int(n * min(1, t / max(0.4, min(1.0, (sc['end'] - sc['start']) * 0.5)))) + 1
            draw_rich(img, sc['text'], W / 2, 930, 92, maxw=940, chars=chars)
        self.caption(img, i, T)
        pr = min(1, T / self.total); d = ImageDraw.Draw(img); d.rectangle([0, 0, int(W * pr), 8], fill=RED)
        return img.convert('RGB')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('out')
    ap.add_argument('--tts', default='tts'); ap.add_argument('--logo', default='logo.png'); ap.add_argument('--maple', default=None)
    ap.add_argument('--stills', nargs='*', type=float)
    a = ap.parse_args(); spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); name = spec['asset_name']
    voice = make_voice(spec, a.tts, out)
    scenes, vo, words, total = plan(spec, voice)
    P = Painter(scenes, words, total, a.logo, a.maple)
    if a.stills is not None:
        for T in a.stills: P.frame(T).save(out / f'still_{T:05.2f}.jpg', quality=85)
        print('total', round(total, 2)); return
    soundtrack(spec, scenes, vo, total, str(out / 'mix.wav'))
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-i', str(out / 'mix.wav'), '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                            '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '160k',
                            '-ar', '48000', '-shortest', '-movflags', '+faststart', str(out / f'{name}.mp4')], stdin=subprocess.PIPE)
    for k in range(int(round(total * FPS))): enc.stdin.write(P.frame(k / FPS).tobytes())
    enc.stdin.close(); enc.wait()
    P.frame(min(0.9, scenes[0]['end'] - 0.1)).save(out / f'{name}_cover.jpg', quality=92)
    (out / 'mix.wav').unlink(missing_ok=True)
    print(json.dumps({'video': str(out / f'{name}.mp4'), 'cover': str(out / f'{name}_cover.jpg'), 'seconds': round(total, 2)}))

if __name__ == '__main__':
    main()
