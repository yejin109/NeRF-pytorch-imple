import subprocess
import datetime


def get_parent_dir(path):
    res = "/".join(path.split('/')[:-1])
    return res


def run_cmd(line):
    out = subprocess.run(line, capture_output=True, shell=True, check=False)
    res = f'\n Out : {out.stdout.decode("cp949")} \n Err : {out.stderr.decode("cp949")}'
    return res


def log(txt, add_time=True):
    if add_time:
        txt = f'{str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))} : {txt}'
    with open("./logs/debug/[pkg]preprocess.txt", 'a') as f:
        f.write(f"{txt}\n")
    f.close()
