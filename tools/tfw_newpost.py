"""TheFrenchWay2PR "new post" story (1080x1920): announces a feed post, using its first slide.

Usage: python3 tfw_newpost.py COVER.jpg OUT.png --sub "25 article mistakes to fix" [--button "See it on our feed"] [--logo logo.png]
(default button "Swipe it on our feed" suits carousels; use "See it on our feed" for a single image)
Needs tfw_template.py and tfw_story.py in the same folder.
"""
import argparse, asyncio, base64
from pathlib import Path
from playwright.async_api import async_playwright
import sys; sys.path.insert(0, str(Path(__file__).parent))
from tfw_template import fmt
from tfw_story import shell

EXTRA = """
.np{position:absolute;top:380px;left:64px;right:64px;text-align:center}
.np h1{font-size:80px;font-weight:800;line-height:1.02}
.np .sub{font-size:32px;color:#C9D5F5;margin-top:12px}
.frame{position:absolute;top:590px;left:50%;width:700px;margin-left:-350px;transform:rotate(-2.5deg);
 border-radius:30px;overflow:hidden;box-shadow:0 0 0 8px #fff,0 30px 70px rgba(0,0,0,.5)}
.frame img{display:block;width:100%}
.go{position:absolute;left:64px;right:64px;bottom:160px;text-align:center}
.go .b{display:inline-block;background:#E1251B;padding:20px 40px;border-radius:44px;font-size:36px;font-weight:800;box-shadow:0 12px 30px rgba(0,0,0,.35)}
.go .s{font-size:26px;color:#DCE6FF;margin-top:16px;font-weight:500}
"""

async def main(a):
    logo = "data:image/png;base64," + base64.b64encode(Path(a.logo).read_bytes()).decode()
    cover = "data:image/jpeg;base64," + base64.b64encode(Path(a.cover).read_bytes()).decode()
    inner = (f'<style>{EXTRA}</style><div class="np"><h1>Nouveau post&nbsp;!</h1>'
             f'<div class="sub">{fmt(a.sub)}</div></div>'
             f'<div class="frame"><img src="{cover}"></div>'
             f'<div class="go"><div class="b" style="display:inline-block">{fmt(a.button)}</div><div class="s">Save it for later · Enregistre-le</div></div>')
    html = shell(inner, logo, "NEW POST")
    async with async_playwright() as p:
        br = await p.chromium.launch(); pg = await br.new_page(viewport={"width": 1080, "height": 1920})
        await pg.set_content(html); await pg.wait_for_timeout(300)
        await pg.screenshot(path=a.out); await br.close()
    print(a.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cover"); ap.add_argument("out")
    ap.add_argument("--sub", default="Now on our feed"); ap.add_argument("--logo", default="logo.png")
    ap.add_argument("--button", default="Swipe it on our feed")
    asyncio.run(main(ap.parse_args()))
