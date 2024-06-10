import datetime

def log(str):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"({time_str}) {str}")

def logV2(job_id, str):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"({time_str}) [job {job_id}] {str}")

def error(job_id, str):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"({time_str}) [job {job_id}] ERROR: {str}")
    raise str
