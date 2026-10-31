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
BPM = {'funny': 124, 'mistakes': 120, 'technique': 114, 'serious': 100}

def plan(spec, voice):
    """scene times; every cut lands on a beat of the music"""
    beat = 60 / BPM.get(spec.get('style'), 118)
    pace = {'funny': 0.15, 'mistakes': 0.18, 'technique': 0.24, 'serious': 0.28}.get(spec.get('style'), 0.2)
    t, scenes, vo, words = 0.0, [], [], []
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
            v = st + {'hook': 0.5, 'cta': 1.4, 'wrong': 1.3, 'right': 1.2}.get(sc['type'], 0.8) + (0.2 if sc['type'] == 'hook' else 0.27) * n
            if sc['type'] == 'tip': v += 0.3 * len(plain(sc.get('text', ''))) / 30
        hold = {'cta': 1.4, 'right': 0.5, 'wrong': 0.35, 'hook': 0.1}.get(sc['type'], pace)
        end = max(v - 0.1 + hold, st + {'hook': 1.6}.get(sc['type'], 1.0))
        end = math.ceil(end / beat - 0.25) * beat  # snap the cut to the beat
        scenes.append({**sc, 'start': st, 'end': end}); t = end
    return scenes, vo, words, t

# ---------------- audio ----------------
def soundtrack(spec, scenes, vo, total, path):
    """upbeat synth track (four-on-the-floor, pumping bass, plucked arpeggio), drop after the hook, sound effects"""
    N = int((total + 0.3) * SR); rng = np.random.default_rng(7)
    def add(buf, st, sig):
        i = int(round(st * SR))
        if i < 0: sig = sig[-i:]; i = 0
        if i < len(buf): j = min(len(buf), i + len(sig)); buf[i:j] += sig[:j - i]
    lp = lambda x, f: sosfilt(butter(2, f, 'low', fs=SR, output='sos'), x)
    hp = lambda x, f: sosfilt(butter(2, f, 'high', fs=SR, output='sos'), x)
    bp = lambda x, a, b: sosfilt(butter(2, [a, b], 'band', fs=SR, output='sos'), x)
    midi = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    env = lambda n, k: np.exp(-np.arange(n) / SR * k)
    def saw(f, n, det=(0,)):
        tt = np.arange(n) / SR; return sum(2 * ((tt * f * 2 ** (c / 1200) + rng.random()) % 1) - 1 for c in det) / len(det)
    style = spec.get('style', 'mistakes'); bpm = BPM.get(style, 118); beat = 60 / bpm; bar = 4 * beat; s16 = beat / 4
    prog = {'serious': [[45, 57, 60, 64], [41, 57, 60, 65], [48, 55, 60, 64], [43, 55, 59, 62]],
            'funny':   [[48, 60, 64, 67], [53, 60, 65, 69], [45, 60, 64, 69], [55, 59, 62, 67]]}.get(
        style, [[45, 57, 60, 64], [41, 57, 60, 65], [48, 55, 60, 67], [43, 55, 59, 62]])  # Am F C G
    drop = scenes[0]['end']
    stops = [sc['start'] - beat for sc in scenes if sc['type'] == 'right']  # one beat of silence before the answer
    in_stop = lambda x: any(s0 <= x < s0 + beat - 0.01 for s0 in stops)
    drums, bass, pad, arp = (np.zeros(N) for _ in range(4))
    nb = int(total / beat) + 2
    for k in range(nb):  # drums
        st = k * beat
        if st < drop - 1e-3 or in_stop(st): continue
        n = int(0.4 * SR); tt = np.arange(n) / SR
        if style != 'serious' or k % 2 == 0:
            add(drums, st, np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-tt * 28)) / SR) * env(n, 9) * 0.95)
        if k % 2 == 1:  # clap
            for o in (0, 0.011, 0.023):
                m = int(0.18 * SR); add(drums, st + o, bp(rng.standard_normal(m), 900, 3200) * env(m, 30 if o < 0.02 else 14) * 0.22)
        m = int(0.12 * SR); add(drums, st + beat / 2, hp(rng.standard_normal(m), 7000) * env(m, 22) * 0.10)  # open hat
        for q in (1, 3):
            m = int(0.03 * SR); add(drums, st + q * s16, hp(rng.standard_normal(m), 9000) * env(m, 120) * 0.05)
    for b in range(int(total / bar) + 2):  # harmony
        ch = prog[b % 4]; st = b * bar
        n = int(bar * SR) + 2000
        pad_sig = sum(saw(midi(m), n, (-9, 0, 9)) for m in ch[1:]) / 3
        add(pad, st, pad_sig * np.minimum(1, np.arange(n) / SR * 8) * 0.10)
        for e in range(8):  # pumping offbeat bass
            n2 = int(beat / 2 * SR * 0.9); x = saw(midi(ch[0] - 12), n2, (-6, 6)) + 0.5 * np.sin(2 * np.pi * midi(ch[0] - 24) * np.arange(n2) / SR)
            if st + e * beat / 2 >= drop and not in_stop(st + e * beat / 2):
                add(bass, st + e * beat / 2 + (beat / 2 if style != 'serious' else 0) * 0, lp(x, 700) * np.minimum(1, np.arange(n2) / SR * 200) * env(n2, 4) * (0.33 if e % 2 else 0.2))
        notes = [m + 12 for m in ch[1:]] + [ch[1] + 24]
        pat = {'funny': [0, 2, 1, 3, 0, 2, 3, 1], 'serious': [0, 1, 2, 1]}.get(style, [0, 1, 2, 3, 2, 1, 3, 2])
        step = s16 * (2 if style == 'serious' else 1) * (2 if style == 'funny' else 1)
        for q in range(int(bar / step)):
            if style == 'funny' and q % 4 == 3: continue
            m = notes[pat[q % len(pat)]]; n3 = int(0.3 * SR); tt = np.arange(n3) / SR
            if style == 'funny':  # marimba
                x = (np.sin(2 * np.pi * midi(m) * tt) + 0.25 * np.sin(2 * np.pi * 4 * midi(m) * tt) * env(n3, 60)) * env(n3, 16)
            else:
                x = lp(np.sign(np.sin(2 * np.pi * midi(m) * tt)) * 0.6 + saw(midi(m), n3) * 0.4, 3500) * env(n3, 20)
            add(arp, st + q * step, x * 0.11)
    tt = np.arange(N) / SR  # sidechain pump from the kick
    ph = (tt % beat) / beat; pump = np.where(tt >= drop, 1 - 0.75 * np.exp(-ph * 9), 1.0)
    intro = np.clip(tt / max(drop, 0.1), 0, 1)
    arp_f = np.where(tt < drop, lp(arp, 900) * (0.6 + 0.4 * intro) , arp)  # filtered until the drop
    ir = rng.standard_normal(int(0.9 * SR)) * env(int(0.9 * SR), 5); ir /= np.abs(ir).sum() / 4
    music = drums + (bass + pad) * pump + arp_f * (0.7 + 0.3 * pump) + fftconvolve(arp_f + pad * 0.5, ir)[:N] * 0.12 * pump
    music /= np.abs(music).max() + 1e-9
    sfx = np.zeros(N)
    n = int(0.6 * SR); t6 = np.arange(n) / SR  # hook impact
    add(sfx, 0.0, np.sin(2 * np.pi * np.cumsum(40 + 120 * np.exp(-t6 * 20)) / SR) * env(n, 6) * 0.7 + lp(rng.standard_normal(n), 3000) * env(n, 12) * 0.25)
    n = int(drop * SR)  # riser into the drop
    if n > SR * 0.5:
        x = rng.standard_normal(n); w = np.linspace(0, 1, n) ** 2
        add(sfx, 0, hp(x, 1500) * w * 0.12)
    n = int(1.6 * SR); add(sfx, drop, hp(rng.standard_normal(n), 5000) * env(n, 3) * 0.16)  # crash on the drop
    for s0 in stops:  # riser during the stop
        n = int(beat * SR); tsw = np.arange(n) / SR
        add(sfx, s0, np.sin(2 * np.pi * np.cumsum(300 + 900 * (tsw / beat) ** 2) / SR) * (tsw / beat) * 0.10)
    for sc in scenes[1:]:
        s = sc['start']
        n = int(0.28 * SR); ts = np.arange(n) / SR  # whoosh on every cut
        add(sfx, s - 0.2, bp(rng.standard_normal(n), 400, 4000) * np.sin(np.pi * ts / 0.28) ** 2 * 0.16)
        if sc['type'] == 'wrong':  # buzzer
            n = int(0.45 * SR); ts = np.arange(n) / SR
            add(sfx, s + 0.15, lp(np.sign(np.sin(2 * np.pi * 110 * ts)) + np.sign(np.sin(2 * np.pi * 116 * ts)), 2500) * np.minimum(1, (0.45 - ts) * 20) * 0.10)
        elif sc['type'] == 'right':  # bright chime
            for i, f in enumerate((1047, 1319, 1568, 2093)):
                n = int(0.5 * SR); ts = np.arange(n) / SR; add(sfx, s + 0.05 + i * 0.06, np.sin(2 * np.pi * f * ts) * env(n, 9) * 0.10)
        elif sc['type'] == 'chat':  # bubble pop
            n = int(0.1 * SR); ts = np.arange(n) / SR; add(sfx, s + 0.05, np.sin(2 * np.pi * np.cumsum(400 + 1600 * ts / 0.1) / SR) * env(n, 40) * 0.2)
        elif sc['type'] == 'cta':
            for i, f in enumerate((784, 1175)):
                n = int(0.6 * SR); ts = np.arange(n) / SR; add(sfx, s + 0.1 + i * 0.12, np.sin(2 * np.pi * f * ts) * env(n, 7) * 0.10)
    voice, act = np.zeros(N), np.zeros(N)
    for st, a in vo:
        add(voice, st, hp(a, 80)); act[int(max(0, st - 0.1) * SR):int((st + len(a) / SR + 0.1) * SR)] = 1
    k = int(0.2 * SR); act = np.convolve(act, np.ones(k) / k, 'same')
    mix = music * (0.3 if vo else 0.62) * (1 - 0.5 * act) + sfx + voice
    f = int(0.8 * SR); mix[-f:] *= np.linspace(1, 0, f)
    mix = np.tanh(mix * 1.6) / np.tanh(1.6)  # gentle glue
    sf.write(path, np.stack([mix, mix], 1) / (np.abs(mix).max() + 1e-9) * 0.95, SR)

# ---------------- pictures ----------------
YEL = (255, 214, 64)
# background + text colours per scene type: (top, bottom, text, [b], [r], stripe)
THEMES = {
    'hook':  ((214, 30, 22), (150, 14, 20), WHITE, YEL, YEL, (255, 255, 255, 18)),
    'wrong': ((48, 14, 30), (14, 8, 28), WHITE, LBLUE, LRED, (255, 255, 255, 10)),
    'right': ((0, 96, 186), (8, 34, 104), WHITE, YEL, LRED, (255, 255, 255, 14)),
    'tip':   ((252, 248, 238), (236, 228, 210), INK, BLUE, RED, (11, 28, 74, 12)),
    'chat':  ((20, 46, 110), (7, 20, 54), WHITE, LBLUE, LRED, (255, 255, 255, 10)),
    'cta':   ((214, 30, 22), (120, 10, 18), WHITE, YEL, YEL, (255, 255, 255, 18)),
}
EMOJI = {'hook': {'funny': '😂', 'technique': '🧠', 'serious': '⚠️'}, 'wrong': '😬', 'right': '🎉', 'tip': '💡', 'cta': '🔔'}
_EMO = {}
def emoji(ch, size):
    key = (ch, size)
    if key not in _EMO:
        try:
            f = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf', 109)
            im = Image.new('RGBA', (160, 160), (0, 0, 0, 0)); ImageDraw.Draw(im).text((80, 80), ch, font=f, embedded_color=True, anchor='mm')
            bb = im.getbbox(); im = im.crop(bb) if bb else im
            _EMO[key] = im.resize((size, int(size * im.size[1] / im.size[0])), Image.LANCZOS)
        except Exception: _EMO[key] = None
    return _EMO[key]

class Painter:
    def __init__(self, scenes, words, total, logo_path, maple, style='mistakes'):
        self.scenes, self.words, self.total, self.style = scenes, words, total, style
        self.beat = 60 / BPM.get(style, 118); self.drop = scenes[0]['end']
        logo = Image.open(logo_path).convert('RGBA'); m = Image.new('L', logo.size, 0)
        ImageDraw.Draw(m).ellipse([0, 0, logo.size[0] - 1, logo.size[1] - 1], fill=255); logo.putalpha(m); self.logo = logo
        self.small_logo = logo.resize((96, 96), Image.LANCZOS)
        self.period = 160; self.bgs = {}
        for k, (top, bot, *_ , stripe) in THEMES.items():
            yy = np.linspace(0, 1, H + self.period)[:, None, None]
            g = np.array(top) * (1 - yy) + np.array(bot) * yy
            im = Image.fromarray(np.repeat(np.clip(g, 0, 255).astype(np.uint8), W, 1)).convert('RGBA')
            ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            for x in range(-H, W + H, self.period):  # diagonal stripes
                od.polygon([(x, 0), (x + 60, 0), (x + 60 + H + self.period, H + self.period), (x + H + self.period, H + self.period)], fill=stripe)
            im.alpha_composite(ov); self.bgs[k] = im
        gl = Image.new('RGBA', (1000, 1000), (0, 0, 0, 0)); ImageDraw.Draw(gl).ellipse([200, 200, 800, 800], fill=(255, 255, 255, 40))
        self.glow = gl.filter(ImageFilter.GaussianBlur(130))
        r = np.random.default_rng(3)  # confetti
        self.conf = [(r.uniform(-900, 900), r.uniform(-1900, -900), r.uniform(0, 6.28), r.uniform(-12, 12),
                      [RED, WHITE, YEL, (90, 160, 255)][r.integers(4)], r.uniform(14, 26)) for _ in range(90)]

    def theme(self, typ): return THEMES.get(typ, THEMES['tip'])

    def sticker(self, img, sc, t, x, y, size=150):
        ch = sc.get('emoji', None)
        if ch is None:
            e = EMOJI.get(sc['type']); ch = e.get(self.style, '🚨') if isinstance(e, dict) else e
        if not ch: return
        im = emoji(ch, size)
        if im is None: return
        s = back((t - 0.15) / 0.35)
        if s <= 0.01: return
        im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.BILINEAR).rotate(12 * math.sin(t * 6), expand=True, resample=Image.BICUBIC)
        img.alpha_composite(im, (int(x - im.size[0] / 2), int(y - im.size[1] / 2)))

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
                    d.text((x, 1500), w[1], font=f, fill=YEL if on else WHITE, anchor='lm'); x += d.textlength(w[1] + ' ', font=f)
                return

    def card(self, w, h, fill=CREAM):
        c = Image.new('RGBA', (w + 60, h + 60), (0, 0, 0, 0)); d = ImageDraw.Draw(c)
        d.rounded_rectangle([30, 44, w + 30, h + 44], 36, fill=(0, 0, 0, 110)); c = c.filter(ImageFilter.GaussianBlur(14))
        ImageDraw.Draw(c).rounded_rectangle([30, 30, w + 30, h + 30], 36, fill=fill); return c

    def words_shown(self, text, t, per=0.11):
        """word-by-word reveal: how many characters are visible at time t"""
        p = plain(text); idx = [m.end() for m in re.finditer(r'\S+', p)]
        k = min(len(idx), int(t / per) + 1); return idx[k - 1] if idx else 0

    def scene(self, i, t, T):
        sc = self.scenes[i]; typ = sc['type']; top, bot, base, cb, cr, _ = self.theme(typ)
        off = int((T * 70) % self.period); img = self.bgs[typ if typ in THEMES else 'tip'].crop((0, off, W, off + H))
        gx, gy = W * (0.5 + 0.3 * math.sin(T * 0.9)), H * (0.45 + 0.1 * math.cos(T * 0.7))
        img.alpha_composite(self.glow, (int(gx - 500), int(gy - 500)))
        d = ImageDraw.Draw(img)
        if typ == 'hook':
            lay = Image.new('RGBA', (W, 900), (0, 0, 0, 0))
            draw_rich(lay, sc['text'], W / 2, 470, 128, base=base, b=cb, r=cr, maxw=980, chars=self.words_shown(sc['text'], t, 0.09))
            if sc.get('badge'):
                ld = ImageDraw.Draw(lay); f = font(46, 'ExtraBold'); bw = ld.textlength(sc['badge'].upper(), font=f) + 70
                ld.rounded_rectangle([(W - bw) / 2, 110, (W + bw) / 2, 200], 45, fill=WHITE + (255,))
                ld.text((W / 2, 155), sc['badge'].upper(), font=f, fill=RED, anchor='mm')
            s = back(t / 0.3); shake = int(14 * math.sin(t * 70) * max(0, 1 - t / 0.3))
            lay = lay.resize((max(1, int(W * s)), max(1, int(900 * s))), Image.BILINEAR)
            img.alpha_composite(lay, (int((W - lay.size[0]) / 2) + shake, int(880 - lay.size[1] / 2)))
            self.sticker(img, sc, t, W / 2, 1400, 200)
        elif typ in ('wrong', 'right'):
            a = back(t / 0.35); bad = typ == 'wrong'
            c = self.card(900, 520); cd = ImageDraw.Draw(c); col = RED if bad else GREEN
            cd.ellipse([80, 80, 200, 200], fill=col)
            dj = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
            cd.text((140, 138), '✗' if bad else '✓', font=ImageFont.truetype(dj, 76) if Path(dj).exists() else font(70), fill=WHITE, anchor='mm')
            cd.text((230, 140), 'Not natural' if bad else 'Say this', font=font(44, 'Bold'), fill=col, anchor='lm')
            sz = draw_rich(c, sc['text'], 480, 330, 84, base=INK, b=BLUE, r=RED, maxw=800)
            if bad:
                p = ease((t - 0.4) / 0.3)
                if p > 0:
                    tw = min(800, cd.textlength(plain(sc['text']).split('\n')[0], font=font(sz, 'ExtraBold')))
                    cd.line([(480 - tw / 2, 330), (480 - tw / 2 + tw * p, 330)], fill=RED, width=14)
            if sc.get('sub'): cd.text((480, 470), sc['sub'], font=font(40, 'Medium'), fill=(80, 88, 120), anchor='mm')
            s = max(0.01, a); c = c.resize((max(1, int(c.size[0] * s)), max(1, int(c.size[1] * s))), Image.BILINEAR)
            if bad: c = c.rotate(-3 * min(1, t / 0.5), expand=True, resample=Image.BICUBIC)
            img.alpha_composite(c, (int((W - c.size[0]) / 2), int(870 - c.size[1] / 2)))
            title = 'WRONG ❌' if bad else 'RIGHT ✅'
            f = font(110, 'ExtraBold'); ta = ease(t / 0.2)
            d.text((W / 2, 470 - 40 * (1 - ta)), 'WRONG' if bad else 'RIGHT', font=f, fill=(LRED if bad else YEL), anchor='mm')
            self.sticker(img, sc, t, W / 2 + 380, 600, 170)
            if not bad:  # confetti
                for x0, vy, rot, vr, colr, sz2 in self.conf:
                    tt = t * 1.0; x = W / 2 + x0 * tt; y = 900 + vy * tt + 2600 * tt * tt
                    if tt <= 0 or y > H + 50: continue
                    ang = rot + vr * tt; dx, dy = math.cos(ang) * sz2, math.sin(ang) * sz2 * 0.5
                    d.polygon([(x - dx, y - dy), (x + dy, y - dx), (x + dx, y + dy), (x - dy, y + dx)], fill=colr)
        elif typ == 'chat':
            a = back(t / 0.3)
            prev = [s for s in self.scenes[:i] if s['type'] == 'chat'][-2:]
            y0 = 520
            for k, s2 in enumerate(prev + [sc]):
                l2 = s2.get('who', 'left') == 'left'; cur = s2 is sc; sa = a if cur else 1
                f = font(56, 'Bold'); txt = plain(s2['text']); tw = min(760, d.textlength(txt, font=f) + 80)
                bub = Image.new('RGBA', (int(tw) + 20, 170), (0, 0, 0, 0)); bd = ImageDraw.Draw(bub)
                bd.rounded_rectangle([10, 10, tw + 10, 160], 50, fill=(WHITE if l2 else BLUE) + (255,))
                draw_rich(bub, s2['text'], (tw + 20) / 2, 85, 56, base=INK if l2 else WHITE, b=BLUE if l2 else YEL, r=RED, weight='Bold', maxw=tw - 60)
                bub = bub.resize((max(1, int(bub.size[0] * sa)), max(1, int(bub.size[1] * sa))), Image.BILINEAR)
                av = (90 if l2 else W - 90); ay = y0 + k * 260 + 80
                d.ellipse([av - 55, ay - 55, av + 55, ay + 55], fill=RED if l2 else (255, 196, 64))
                d.text((av, ay), (s2.get('name') or ('A' if l2 else 'B'))[0].upper(), font=font(56, 'ExtraBold'), fill=WHITE, anchor='mm')
                d.text((av, ay + 80), s2.get('name', ''), font=font(28, 'SemiBold'), fill=(210, 222, 255), anchor='mm')
                bx = 170 if l2 else W - 170 - bub.size[0]; img.alpha_composite(bub, (int(bx), int(ay - bub.size[1] / 2)))
        elif typ == 'cta':
            a = back(t / 0.45); sz = max(1, int(400 * a)); lg = self.logo.resize((sz, sz), Image.BILINEAR)
            img.alpha_composite(lg, (int((W - sz) / 2), int(680 - sz / 2)))
            draw_rich(img, sc['text'], W / 2, 1060, 88, base=base, b=cb, r=cr, maxw=940, chars=self.words_shown(sc['text'], t - 0.2, 0.1) if t > 0.2 else 0)
            fa = ease((t - 0.6) / 0.25)
            if fa > 0:
                p = 1 + 0.05 * math.sin(t * 8); tw = 600 * p; hh = 56 * p
                d.rounded_rectangle([(W - tw) / 2, 1250 - hh, (W + tw) / 2, 1250 + hh], int(hh), fill=WHITE)
                d.text((W / 2, 1250), 'Follow @thefrenchway2pr', font=font(40, 'ExtraBold'), fill=RED, anchor='mm')
            self.sticker(img, sc, t, W / 2 + 300, 520, 150)
        else:  # tip
            if sc.get('label'):
                f = font(44, 'ExtraBold'); lw = d.textlength(sc['label'].upper(), font=f) + 70; a = back(t / 0.3)
                pill = Image.new('RGBA', (int(lw), 96), (0, 0, 0, 0)); pd = ImageDraw.Draw(pill)
                pd.rounded_rectangle([0, 0, lw - 1, 95], 48, fill=RED); pd.text((lw / 2, 48), sc['label'].upper(), font=f, fill=WHITE, anchor='mm')
                pill = pill.resize((max(1, int(lw * a)), max(1, int(96 * a))), Image.BILINEAR).rotate(-4, expand=True, resample=Image.BICUBIC)
                img.alpha_composite(pill, (int((W - pill.size[0]) / 2), int(600 - pill.size[1] / 2)))
            draw_rich(img, sc['text'], W / 2, 950, 96, base=base, b=cb, r=cr, maxw=940, chars=self.words_shown(sc['text'], t, 0.12))
            self.sticker(img, sc, t, W / 2, 1380, 170)
        return img

    def frame(self, T):
        for i, sc in enumerate(self.scenes):
            if sc['start'] <= T < sc['end'] or i == len(self.scenes) - 1: break
        t = T - sc['start']; dur = sc['end'] - sc['start']
        img = self.scene(i, t, T)
        TR = 0.2
        if i > 0 and t < TR:  # whip-pan push from the right
            p = ease(t / TR); prev = self.scene(i - 1, self.scenes[i - 1]['end'] - self.scenes[i - 1]['start'], T)
            both = Image.new('RGBA', (W, H)); both.paste(prev, (int(-W * p), 0)); both.paste(img, (int(W * (1 - p)), 0))
            img = both.filter(ImageFilter.BoxBlur(int(10 * math.sin(math.pi * p)))) if p < 0.95 else both
        # camera: slow push-in + a punch on every beat after the drop
        z = 1.02 + 0.05 * min(1, t / max(dur, 0.1))
        if T >= self.drop: z += 0.022 * math.exp(-((T % self.beat) / self.beat) * 7)
        sx = sy = 0
        if sc['type'] == 'wrong' and 0.5 < t < 0.85: sx = int(16 * math.sin(t * 90)); sy = int(8 * math.cos(t * 70))
        cw, ch = W / z, H / z; x0, y0 = min(W - cw, max(0, (W - cw) / 2 + sx)), min(H - ch, max(0, (H - ch) / 2 + sy))
        img = img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        if sc['type'] == 'wrong' and 0.5 < t < 0.62: img.alpha_composite(Image.new('RGBA', (W, H), (225, 37, 27, 70)))
        if i == 0 and t < 0.1: img.alpha_composite(Image.new('RGBA', (W, H), (255, 255, 255, int(200 * (1 - t / 0.1)))))
        self.caption(img, i, T)
        d = ImageDraw.Draw(img)  # fixed overlay: handle, flag strip, progress
        dark = sc['type'] == 'tip'
        img.alpha_composite(self.small_logo, (60, 200))
        d.text((172, 248), '@thefrenchway2pr', font=font(30, 'SemiBold'), fill=(40, 52, 90) if dark else (235, 240, 255), anchor='lm')
        for k, c in enumerate([BLUE, WHITE, RED]): d.rectangle([k * W // 3, H - 14, (k + 1) * W // 3, H], fill=c)
        pr = min(1, T / self.total); d.rectangle([0, 0, W, 10], fill=(0, 0, 0, 60)); d.rectangle([0, 0, int(W * pr), 10], fill=YEL)
        return img.convert('RGB')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('out')
    ap.add_argument('--tts', default='tts'); ap.add_argument('--logo', default='logo.png'); ap.add_argument('--maple', default=None)
    ap.add_argument('--stills', nargs='*', type=float)
    a = ap.parse_args(); spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); name = spec['asset_name']
    voice = make_voice(spec, a.tts, out)
    scenes, vo, words, total = plan(spec, voice)
    P = Painter(scenes, words, total, a.logo, a.maple, spec.get('style', 'mistakes'))
    if a.stills is not None:
        for T in a.stills: P.frame(T).save(out / f'still_{T:05.2f}.jpg', quality=85)
        print('total', round(total, 2), [(s['type'], round(s['start'], 2)) for s in scenes]); return
    soundtrack(spec, scenes, vo, total, str(out / 'mix.wav'))
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-i', str(out / 'mix.wav'), '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                            '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '160k',
                            '-ar', '48000', '-shortest', '-movflags', '+faststart', str(out / f'{name}.mp4')], stdin=subprocess.PIPE)
    for k in range(int(round(total * FPS))): enc.stdin.write(P.frame(k / FPS).tobytes())
    enc.stdin.close(); enc.wait()
    P.frame(min(1.0, scenes[0]['end'] - 0.1)).save(out / f'{name}_cover.jpg', quality=92)
    (out / 'mix.wav').unlink(missing_ok=True)
    print(json.dumps({'video': str(out / f'{name}.mp4'), 'cover': str(out / f'{name}_cover.jpg'), 'seconds': round(total, 2)}))

if __name__ == '__main__':
    main()
