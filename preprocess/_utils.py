import subprocess


def get_parent_dir(path):
    res = "/".join(path.split('/')[:-1])
    return res


def run_cmd(line):
    print(line)
    out = subprocess.run(line, capture_output=True, shell=True, check=False)
    out.stdout.decode("utf-8")
    return out
