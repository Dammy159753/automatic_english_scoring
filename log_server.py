# coding=utf-8
import os
from logserver import LogServer


def get_contener_id():
    cmd = "cat /proc/self/cgroup"
    import os
    output = os.popen(cmd)
    rests = output.readlines()
    container_message= rests[-1]
    if not container_message:
        container_id = "no_found"
    else:
        if container_message.find("docker-") != -1:
            container_id = container_message.strip().split("docker-")[-1][:12]
        else:
            container_id = container_message.strip().split("docker/")[-1][:12]
    print("container_id is:", container_id)
    return container_id


filename = get_contener_id()

log_path = "./log_aes/{}".format(filename)
if not os.path.isdir(log_path):
    os.makedirs(log_path)

log_server = LogServer(app='aes', log_path=log_path)

log_server.re_configure_logging(str(filename) + "_log.txt")
print("logfile is: ", str(filename) + "_log.txt")
log_server.logging("logfile is %s" % (str(filename) + "_log.txt"))
