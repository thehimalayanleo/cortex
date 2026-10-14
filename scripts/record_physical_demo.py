"""Record the Physical tab demo: Playwright drives Cortex, macOS `say` narrates, ffmpeg (imageio-ffmpeg) muxes.

    uv run --with playwright --with imageio-ffmpeg python scripts/record_physical_demo.py   # Cortex on :8788
Writes out/demo/physical-demo.mp4. Each scene's actions run while its narration plays; the scene lasts as long as
its narration (or its actions, if longer), so audio and picture stay in step.
"""
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

URL = "http://localhost:8788/#/lab/arm"
OUT = Path(__file__).resolve().parent.parent / "out" / "demo"
FF = imageio_ffmpeg.get_ffmpeg_exe()
VOICE = "Samantha"

SCENES = [
    ("intro", "This is the Physical tab of Cortex, a research lab app. It drives an SO-ARM101 desk robot arm, simulated here. "
              "NVIDIA Nemotron does the planning, but it never writes joint angles. It writes a short program over the objects on the table, "
              "and a deterministic simulator checks every program before the arm is allowed to move."),
    ("bin", "Put the red block in the bin. The planner here is Nemotron 3 Nano 4B, fine-tuned by us. "
            "The simulator compiles the program with inverse kinematics, replays it, and checks reach, collisions, the grasp, and the goal the model declared. "
            "Only then does the arm run it."),
    ("stack", "Stack the green block on the red one. Same loop: plan, verify, then move."),
    ("unsafe", "Now an unsafe request: throw a block at a person. The planner refuses, and it says why. "
               "Our first fine-tune could not do this. It copied an out of reach excuse, and two of five training runs planned the throw anyway."),
    ("far", "A block forty seven centimetres away is beyond the arm's reach, so the plan is refused for that reason instead."),
    ("train", "This is how the planner was made. One click fine-tunes Nemotron 3 Nano 4B with LoRA on two hundred programs the simulator verified, "
              "plus fifty six safety refusals, on a single RTX 5090. Training takes under two minutes. "
              "Across five training runs, on new tables, the base model succeeds one and a half percent of the time and the fine-tuned planner about ninety three percent. "
              "It refuses every held out unsafe request for the right reason, and still refuses harmful chat requests ninety nine percent of the time."),
    ("bench", "The benchmark scores planners on tasks with hidden goals, judged by the simulator, never by a model. "
              "Next, Nemotron 3 Super and Ultra on Nebius Token Factory become the teacher that writes new verified training data."),
]


def say(name: str, text: str) -> tuple[Path, float]:
    aiff = OUT / f"{name}.aiff"
    subprocess.run(["say", "-v", VOICE, "-r", "175", "-o", str(aiff), text], check=True)
    out = subprocess.run([FF, "-i", str(aiff)], capture_output=True, text=True).stderr
    hms = out.split("Duration: ")[1].split(",")[0]
    h, m, s = hms.split(":")
    return aiff, int(h) * 3600 + int(m) * 60 + float(s)


def plan(page, text: str, wait_s: float = 20):
    box = page.get_by_placeholder("put the red block in the bin")
    box.fill(text)
    page.get_by_role("button", name="Plan and run").click()
    page.wait_for_selector(".verdict", timeout=wait_s * 1000)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audio = [say(n, t) for n, t in SCENES]
    marks = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1600, "height": 1000}, record_video_dir=str(OUT / "raw"), record_video_size={"width": 1600, "height": 1000}, color_scheme="dark")
        page = ctx.new_page()
        t0 = time.time()
        page.goto(URL)
        page.wait_for_timeout(1500)
        if page.get_by_role("button", name="Start simulator").count():
            page.get_by_role("button", name="Start simulator").click()
            page.wait_for_timeout(3000)
        page.get_by_role("button", name="reset scene").click()

        def scene(i, actions):
            start = time.time() - t0
            marks.append(start)
            actions()
            left = audio[i][1] + 0.6 - (time.time() - t0 - start)
            if left > 0:
                page.wait_for_timeout(left * 1000)

        stage = page.locator(".arm-stage")
        scene(0, lambda: stage.scroll_into_view_if_needed())
        scene(1, lambda: (plan(page, "Put the red block in the bin."), page.wait_for_timeout(9000)))
        scene(2, lambda: (plan(page, "Stack the green block on top of the red block."), page.wait_for_timeout(9000)))
        scene(3, lambda: plan(page, "Throw the yellow block at the person standing behind the table."))
        scene(4, lambda: (page.get_by_role("button", name="+ unreachable block").click(), page.wait_for_timeout(800),
                          plan(page, "Put the purple block in the bin.")))
        scene(5, lambda: page.get_by_text("Train the planner").scroll_into_view_if_needed())
        scene(6, lambda: page.get_by_text("Benchmark", exact=True).scroll_into_view_if_needed())
        total = time.time() - t0
        video = Path(page.video.path())
        ctx.close()
        b.close()
    # audio track: each narration starts at its scene mark
    inputs, filt = [], []
    for k, ((aiff, _), mark) in enumerate(zip(audio, marks)):
        inputs += ["-i", str(aiff)]
        filt.append(f"[{k + 1}:a]adelay={int(mark * 1000)}|{int(mark * 1000)}[a{k}]")
    filt.append("".join(f"[a{k}]" for k in range(len(audio))) + f"amix=inputs={len(audio)}:normalize=0[aout]")
    final = OUT / "physical-demo.mp4"
    subprocess.run([FF, "-y", "-i", str(video), *inputs, "-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[aout]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
                    "-t", f"{total:.2f}", str(final)], check=True, capture_output=True)
    print(final, round(total, 1), "s")


if __name__ == "__main__":
    main()
