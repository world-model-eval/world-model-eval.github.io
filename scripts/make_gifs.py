import os
import pathlib

rollouts_dir = pathlib.Path("rollouts")
for path in rollouts_dir.iterdir():
    if (str(path).endswith("xs.mp4")):
        p1 = str(path)
        p2 = str(path).replace("xs.mp4", "xs_pred.mp4")
        os.system(f"python concat_to_gif.py {p1} {p2} --output {p1.replace('xs.mp4', '.gif')}")

    