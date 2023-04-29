def log(msg):
    with open('logs/debug/pkg_neus_model.txt', 'a') as f:
        f.write(msg)
    f.close()
