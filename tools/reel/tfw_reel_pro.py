"""TheFrenchWay2PR reel maker, "pro" look: same reel.json as tfw_reel.py, edited like a motion designer would.

Usage (same as tfw_reel.py):
  python3 tfw_reel_pro.py reel.json OUT_DIR --tts TTS_DIR --logo logo.png [--stills 0.5 3 8]

What changes against tfw_reel.py:
- one continuous cinematic background (deep navy, slow drifting light, film grain, vignette) instead of a new
  coloured, striped background on every scene;
- type that slides up out of a mask, line by line, with a small kicker label above; the hook is fully readable on frame 1;
- clean cuts: content eases out up and the next scene eases in, no whip pans, no beat punches, no confetti or spinning emoji;
- glass cards for the wrong/right examples, an animated strike-through on the wrong one;
- big burned-in captions with the spoken word highlighted, inside the safe zone, no black box;
- voice-first mix: the music sits well under the voice and ducks harder while someone speaks.
Needs tfw_reel.py next to it (voice, timing and music come from there).
"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import re
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tfw_reel as base
from tfw_reel import W, H, FPS, SR, font, draw_rich, plain, ease, WHITE, RED, LRED, LBLUE, BLUE

YEL = (255, 210, 70)
NAVY_TOP, NAVY_BOT = (14, 32, 84), (5, 12, 36)
MUTED = (178, 192, 228)
SAFE_TOP, SAFE_BOT = 260, 1440

def smooth(x): x = min(1, max(0, x)); return x * x * (3 - 2 * x)
def out_expo(x): x = min(1, max(0, x)); return 1 - 2 ** (-10 * x) if x < 1 else 1

class Pro:
    def __init__(self, scenes, words, total, logo_path):
        self.scenes, self.words, self.total = scenes, words, total
        yy = np.linspace(0, 1, H)[:, None]; xx = np.linspace(-1, 1, W)[None, :]
        g = np.array(NAVY_TOP)[None, None] * (1 - yy[..., None]) + np.array(NAVY_BOT)[None, None] * yy[..., None]
        vig = 1 - 0.45 * np.clip((xx ** 2 * 0.6 + (yy * 2 - 1) ** 2 * 0.5), 0, 1)
        self.bg = Image.fromarray(np.clip(g * vig[..., None], 0, 255).astype(np.uint8)).convert('RGBA')
        def blob(col, r, a):
            im = Image.new('RGBA', (r * 4, r * 4), (0, 0, 0, 0)); ImageDraw.Draw(im).ellipse([r, r, 3 * r, 3 * r], fill=col + (a,))
            return im.filter(ImageFilter.GaussianBlur(r * 0.55))
        self.blobs = [(blob((40, 90, 230), 380, 120), 0.18, 0.30, 0.11, 0.0),
                      (blob((225, 37, 27), 300, 70), 0.82, 0.72, 0.08, 2.0),
                      (blob((255, 255, 255), 260, 28), 0.55, 0.52, 0.06, 4.0)]
        rng = np.random.default_rng(11)
        self.grain = [Image.fromarray(np.dstack([np.full((H // 2, W // 2), 128, np.uint8)] * 3 +
                      [(rng.random((H // 2, W // 2)) * 22).astype(np.uint8)])).resize((W, H), Image.NEAREST) for _ in range(4)]
        logo = Image.open(logo_path).convert('RGBA'); m = Image.new('L', logo.size, 0)
        ImageDraw.Draw(m).ellipse([0, 0, logo.size[0] - 1, logo.size[1] - 1], fill=255); logo.putalpha(m)
        self.logo = logo; self.small_logo = logo.resize((84, 84), Image.LANCZOS)
        self.cache = {}

    # ---- building blocks ----
    def background(self, T):
        img = self.bg.copy()
        for im, cx, cy, sp, ph in self.blobs:
            x = W * (cx + 0.10 * math.sin(T * sp * 2 * math.pi / 3 + ph)); y = H * (cy + 0.06 * math.cos(T * sp * 2 * math.pi / 4 + ph))
            img.alpha_composite(im, (int(x - im.size[0] / 2), int(y - im.size[1] / 2)))
        return img

    def text_lines(self, text, size, base_col=WHITE, b=YEL, r=LRED, weight='ExtraBold', maxw=920):
        """render each line of rich text to its own transparent image (cached)"""
        key = (text, size, base_col, b, r, weight, maxw)
        if key in self.cache: return self.cache[key]
        probe = Image.new('RGBA', (W, 400)); sz = size
        lines = text.split('\n'); d = ImageDraw.Draw(probe)
        while True:  # shrink so every line fits
            f = font(sz, weight)
            if max(d.textlength(plain(l), font=f) for l in lines) <= maxw or sz < 34: break
            sz -= 4
        out = []
        for ln in lines:
            im = Image.new('RGBA', (W, int(sz * 1.45)), (0, 0, 0, 0))
            sh = Image.new('RGBA', im.size, (0, 0, 0, 0))
            draw_rich(sh, ln, W / 2, im.size[1] / 2 + 6, sz, base=(0, 0, 0), b=(0, 0, 0), r=(0, 0, 0), weight=weight, maxw=maxw)
            sh = sh.filter(ImageFilter.GaussianBlur(10)); a = sh.getchannel('A').point(lambda v: v * 0.55); sh.putalpha(a)
            im.alpha_composite(sh)
            draw_rich(im, ln, W / 2, im.size[1] / 2, sz, base=base_col, b=b, r=r, weight=weight, maxw=maxw)
            out.append(im)
        self.cache[key] = (out, sz); return out, sz

    def reveal(self, img, lines, cy, t, delay=0.0, stagger=0.09, dur=0.45, alpha=1.0, instant=False):
        """masked slide-up reveal, line by line, centred on cy"""
        lh = lines[0].size[1] * 0.86; y0 = cy - lh * (len(lines) - 1) / 2
        for k, ln in enumerate(lines):
            p = 1.0 if instant else out_expo((t - delay - k * stagger) / dur)
            if p <= 0: continue
            hgt = ln.size[1]; dy = int((1 - p) * hgt * 0.9)
            piece = ln.crop((0, 0, W, hgt - dy)) if dy > 0 else ln
            if alpha < 1: piece = piece.copy(); piece.putalpha(piece.getchannel('A').point(lambda v: int(v * alpha)))
            img.alpha_composite(piece, (0, int(y0 + k * lh - hgt / 2 + dy)))

    def kicker(self, img, label, y, t, col=LRED, alpha=1.0):
        if not label: return
        d = ImageDraw.Draw(img); f = font(38, 'Bold'); txt = '  '.join(label.upper())  # tracked out
        txt = label.upper(); tw = d.textlength(txt, font=f)
        p = out_expo(t / 0.4); a = int(255 * p * alpha)
        lw = int(70 * p)
        d.line([(W / 2 - tw / 2 - 30 - lw, y), (W / 2 - tw / 2 - 30, y)], fill=col + (a,), width=4)
        d.line([(W / 2 + tw / 2 + 30, y), (W / 2 + tw / 2 + 30 + lw, y)], fill=col + (a,), width=4)
        d.text((W / 2, y + (1 - p) * 20), txt, font=f, fill=col + (a,), anchor='mm')

    def glass(self, w, h, tint=(255, 255, 255), a=26):
        key = ('glass', w, h, tint, a)
        if key not in self.cache:
            c = Image.new('RGBA', (w + 80, h + 80), (0, 0, 0, 0)); d = ImageDraw.Draw(c)
            d.rounded_rectangle([40, 56, w + 40, h + 56], 44, fill=(0, 0, 0, 90)); c = c.filter(ImageFilter.GaussianBlur(20))
            d = ImageDraw.Draw(c)
            d.rounded_rectangle([40, 40, w + 40, h + 40], 44, fill=tint + (a,), outline=(255, 255, 255, 60), width=2)
            self.cache[key] = c
        return self.cache[key]

    def caption(self, img, i, T):
        ws = [w for w in self.words if w[0] == i]
        if not ws: return
        chunks, cur = [], []
        for w in ws:
            cur.append(w)
            if len(cur) == 3 or w[1][-1] in '.,?!:;': chunks.append(cur); cur = []
        if cur: chunks.append(cur)
        for j, ch in enumerate(chunks):
            nxt = chunks[j + 1][0][2] if j + 1 < len(chunks) else ch[-1][3] + 0.3
            if not (ch[0][2] <= T < nxt): continue
            txt = ' '.join(w[1] for w in ch)
            norm = lambda x: re.sub(r'[^\w ]', '', x.lower()).strip()
            if norm(txt) and norm(txt) in norm(plain(self.scenes[i].get('text', '')).replace('\n', ' ')): return  # already on screen
            d = ImageDraw.Draw(img); f = font(60, 'ExtraBold'); tw = d.textlength(txt, font=f); x = (W - tw) / 2; y = 1360
            pin = out_expo((T - ch[0][2]) / 0.18); y += (1 - pin) * 18
            for w in ch:
                on = w[2] <= T < w[3] + 0.05
                for ox, oy in ((0, 5), (2, 6), (-2, 6)): d.text((x + ox, y + oy), w[1], font=f, fill=(0, 0, 0, int(140 * pin)), anchor='lm')
                d.text((x, y), w[1], font=f, fill=(YEL if on else WHITE) + (int(255 * pin),), anchor='lm')
                x += d.textlength(w[1] + ' ', font=f)
            return

    # ---- scenes ----
    def scene(self, img, sc, t, dur):
        typ = sc['type']
        # ease out the last 0.18 s: content fades and lifts, the background keeps moving (clean cut)
        out = smooth((t - (dur - 0.18)) / 0.18); a = 1 - out; lift = int(-50 * out)
        lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        m = re.search(r'\[[br]\](.*?)\[/[br]\]', sc.get('text', ''))
        if m and typ != 'cta' and len(m.group(1).strip()) <= 2:  # the key letters, huge and faint, drifting
            g = m.group(1).strip(); f = font(760, 'ExtraBold'); gp = out_expo(t / 0.8)
            ImageDraw.Draw(lay).text((W / 2 + 40 * math.sin(t * 0.6), 930 + 30 * (1 - gp)), g, font=f, fill=(255, 255, 255, int(16 * gp)), anchor='mm')
        if typ == 'hook':
            lines, _ = self.text_lines(sc['text'], 124, b=YEL, r=YEL)
            s = 1.06 - 0.06 * out_expo(t / 0.6)
            self.kicker(lay, sc.get('badge', ''), 640, t + 0.4, YEL)
            self.reveal(lay, lines, 900, t, instant=True)
            if abs(s - 1) > 1e-3:
                lay = lay.resize((int(W * s), int(H * s)), Image.BILINEAR).crop((int((W * s - W) / 2), int((H * s - H) / 2), int((W * s - W) / 2) + W, int((H * s - H) / 2) + H))
        elif typ in ('wrong', 'right'):
            bad = typ == 'wrong'; col = RED if bad else (34, 170, 100)
            self.kicker(lay, 'Not natural' if bad else 'Say it like this', 560, t, LRED if bad else (110, 220, 150))
            p = out_expo(t / 0.5)
            g = self.glass(880, 440, (255, 80, 70) if bad else (255, 255, 255), 30 if bad else 24)
            gy = int(900 - g.size[1] / 2 + (1 - p) * 60); ga = g.copy(); ga.putalpha(ga.getchannel('A').point(lambda v: int(v * p)))
            lay.alpha_composite(ga, (int((W - g.size[0]) / 2), gy))
            d = ImageDraw.Draw(lay)
            d.ellipse([W / 2 - 40, 735 - 40 + (1 - p) * 60, W / 2 + 40, 735 + 40 + (1 - p) * 60], fill=col + (int(255 * p),))
            dj = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
            gf = ImageFont.truetype(dj, 46) if Path(dj).exists() else font(46, 'Bold')
            d.text((W / 2, 735 + (1 - p) * 60), '✗' if bad else '✓', font=gf, fill=(255, 255, 255, int(255 * p)), anchor='mm')
            lines, sz = self.text_lines(sc['text'], 96, b=YEL, r=LRED, maxw=780)
            self.reveal(lay, lines, 880, t, delay=0.12)
            if bad:  # strike-through draws on after the line is read
                q = smooth((t - 0.7) / 0.35)
                if q > 0:
                    tw = min(780, ImageDraw.Draw(Image.new('RGBA', (1, 1))).textlength(plain(sc['text']).split('\n')[0], font=font(sz, 'ExtraBold')))
                    d.line([(W / 2 - tw / 2 - 10, 884), (W / 2 - tw / 2 - 10 + (tw + 20) * q, 884)], fill=LRED + (255,), width=10)
            if sc.get('sub'):
                sp = out_expo((t - 0.35) / 0.4)
                d.text((W / 2, 1010 + (1 - sp) * 16), sc['sub'], font=font(42, 'Medium'), fill=MUTED + (int(255 * sp),), anchor='mm')
        elif typ == 'chat':
            p = out_expo(t / 0.45); l2 = sc.get('who', 'left') == 'left'
            self.kicker(lay, sc.get('name', ''), 640, t, YEL)
            lines, _ = self.text_lines(sc['text'], 84, maxw=820)
            g = self.glass(900, 260 + 90 * (len(lines) - 1)); gg = g.copy(); gg.putalpha(gg.getchannel('A').point(lambda v: int(v * p)))
            lay.alpha_composite(gg, (int((W - g.size[0]) / 2), int(900 - g.size[1] / 2 + (1 - p) * 50)))
            self.reveal(lay, lines, 900, t, delay=0.1)
        elif typ == 'cta':
            p = out_expo(t / 0.6); sz = int(230 + 20 * p); lg = self.logo.resize((sz, sz), Image.LANCZOS)
            lg.putalpha(lg.getchannel('A').point(lambda v: int(v * p)))
            lay.alpha_composite(lg, (int((W - sz) / 2), int(560 - sz / 2)))
            lines, _ = self.text_lines(sc['text'], 92, b=YEL, r=YEL)
            self.reveal(lay, lines, 900, t, delay=0.2)
            hp = out_expo((t - 0.7) / 0.5)
            if hp > 0:
                d = ImageDraw.Draw(lay); f = font(40, 'Bold'); txt = '@thefrenchway2pr'; tw = d.textlength(txt, font=f) + 90
                d.rounded_rectangle([(W - tw) / 2, 1120 - 44 + (1 - hp) * 20, (W + tw) / 2, 1120 + 44 + (1 - hp) * 20], 44,
                                    fill=(255, 255, 255, int(235 * hp)))
                d.text((W / 2, 1120 + (1 - hp) * 20), txt, font=f, fill=(11, 28, 74, int(255 * hp)), anchor='mm')
        else:  # tip
            self.kicker(lay, sc.get('label', ''), 640, t, LRED)
            lines, _ = self.text_lines(sc['text'], 104, b=YEL, r=LRED)
            self.reveal(lay, lines, 900, t, delay=0.08)
        if a < 1 or lift:
            lay2 = Image.new('RGBA', (W, H), (0, 0, 0, 0)); lay2.paste(lay, (0, lift))
            lay2.putalpha(lay2.getchannel('A').point(lambda v: int(v * a))); lay = lay2
        img.alpha_composite(lay)

    def frame(self, T):
        for i, sc in enumerate(self.scenes):
            if sc['start'] <= T < sc['end'] or i == len(self.scenes) - 1: break
        t = T - sc['start']; dur = sc['end'] - sc['start']
        img = self.background(T)
        self.scene(img, sc, t, dur)
        z = 1.0 + 0.035 * T / max(self.total, 1)  # one slow push-in across the whole reel
        if z > 1.001:
            cw, ch = W / z, H / z; x0, y0 = (W - cw) / 2, (H - ch) / 2
            img = img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
        self.caption(img, i, T)
        img.alpha_composite(self.grain[int(T * FPS) % 4])
        d = ImageDraw.Draw(img)
        img.alpha_composite(self.small_logo, (64, 196))
        d.text((166, 238), '@thefrenchway2pr', font=font(30, 'SemiBold'), fill=(225, 232, 255), anchor='lm')
        pr = min(1, T / self.total); d.rectangle([0, 0, W, 6], fill=(255, 255, 255, 40)); d.rectangle([0, 0, int(W * pr), 6], fill=YEL)
        return img.convert('RGB')

def mix_audio(spec, scenes, vo, total, path, work):
    """music + sound effects from tfw_reel (no buzzer/chime), then a voice-first mix with strong ducking"""
    calm = [{**s, 'type': 'tip' if s['type'] in ('wrong', 'right', 'chat') else s['type']} for s in scenes]
    base.soundtrack(spec, calm, [], total, str(work / 'music.wav'))
    music, _ = sf.read(str(work / 'music.wav')); music = music[:, 0]
    N = len(music); voice, act = np.zeros(N), np.zeros(N)
    for st, a in vo:
        i = int(st * SR); j = min(N, i + len(a)); voice[i:j] += a[:j - i]
        act[max(0, i - int(0.15 * SR)):min(N, j + int(0.2 * SR))] = 1
    k = int(0.25 * SR); act = np.convolve(act, np.ones(k) / k, 'same')
    mix = music * (0.34 if vo else 0.8) * (1 - 0.62 * act) + voice * 1.0
    f = int(0.8 * SR); mix[-f:] *= np.linspace(1, 0, f)
    mix = np.tanh(mix * 1.4) / np.tanh(1.4)
    sf.write(path, np.stack([mix, mix], 1) / (np.abs(mix).max() + 1e-9) * 0.95, SR)
    (work / 'music.wav').unlink(missing_ok=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('out')
    ap.add_argument('--tts', default='tts'); ap.add_argument('--logo', default='logo.png')
    ap.add_argument('--stills', nargs='*', type=float)
    a = ap.parse_args(); spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); name = spec['asset_name']
    voice = base.make_voice(spec, a.tts, out)
    scenes, vo, words, total = base.plan(spec, voice)
    P = Pro(scenes, words, total, a.logo)
    if a.stills is not None:
        for T in a.stills: P.frame(T).save(out / f'still_{T:05.2f}.jpg', quality=88)
        print('total', round(total, 2), [(s['type'], round(s['start'], 2)) for s in scenes]); return
    mix_audio(spec, scenes, vo, total, str(out / 'mix.wav'), out)
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-i', str(out / 'mix.wav'), '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                            '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '192k',
                            '-ar', '48000', '-shortest', '-movflags', '+faststart', str(out / f'{name}.mp4')], stdin=subprocess.PIPE)
    for k in range(int(round(total * FPS))): enc.stdin.write(P.frame(k / FPS).tobytes())
    enc.stdin.close(); enc.wait()
    P.frame(0.9).save(out / f'{name}_cover.jpg', quality=92)
    (out / 'mix.wav').unlink(missing_ok=True)
    print(json.dumps({'video': str(out / f'{name}.mp4'), 'cover': str(out / f'{name}_cover.jpg'), 'seconds': round(total, 2)}))

if __name__ == '__main__':
    main()
