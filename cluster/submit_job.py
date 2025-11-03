from ray.job_submission import JobSubmissionClient, JobStatus
import time

RAY_ADDRESS = "http://10.100.192.238:8265"
client = JobSubmissionClient(RAY_ADDRESS)

# Use -m to run the module
job_id = client.submit_job(
    entrypoint="python3 -m TrackSelectorDNN.tune.tune",
    runtime_env={
        "pip": [
            "-e /eos/user/e/ecoradin/GitHub/TrackSelectorDNN/"  # install package in the job
        ],
    },
)
print(f"Submitted Ray job: {job_id}")

def wait_until_done(job_id, timeout_seconds=3600):
    start = time.time()
    while time.time() - start < timeout_seconds:
        status = client.get_job_status(job_id)
        print(f"Job status: {status}")
        if status in {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}:
            break
        time.sleep(10)
    else:
        print("⚠️ Timeout waiting for job.")
        return

    print("\n--- Job Logs ---")
    logs = client.get_job_logs(job_id)
    print(logs)

    if status == JobStatus.SUCCEEDED:
        print("Job completed successfully!")
    elif status == JobStatus.FAILED:
        print("Job failed.")
    else:
        print("Job stopped manually.")

wait_until_done(job_id)
