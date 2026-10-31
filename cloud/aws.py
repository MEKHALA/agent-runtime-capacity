"""Deploy, run and tear down the platform CPU benchmark on AWS.

    .venv/bin/python cloud/aws.py check      # credentials, region, permissions dry-run
    .venv/bin/python cloud/aws.py deploy     # S3 + IAM roles + Lambda (arm64, x86_64) + AgentCore runtime
    .venv/bin/python cloud/aws.py run        # Lambda at 4 memory sizes x 2 archs, then AgentCore
    .venv/bin/python cloud/aws.py teardown   # deletes everything deploy created

Credentials come from the AWS profile named by $AWS_PROFILE (default
"agentprof"); nothing secret is read or written by this script. Resource
identifiers (bucket name, ARNs, which embed the account ID) are kept in
cloud/.state.json, which is gitignored: never commit it.

No model is ever called. The benchmark fakes the model client's socket, so the
only charges are Lambda and AgentCore compute for a few minutes, plus S3.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "cloud" / ".state.json"
OUT = ROOT / "data" / "platform"
PROFILE = os.environ.get("AWS_PROFILE", "agentprof")
REGION = os.environ.get("AWS_REGION", "us-east-1")
PREFIX = "agentprof-bench"
RUNTIME_NAME = "agentprof_bench"              # AgentCore names allow [a-zA-Z0-9_] only
ARCHS = ("arm64", "x86_64")
LAMBDA_MEMORY_MB = (512, 1024, 1769, 3008)     # 1,769 MB is one full vCPU on Lambda
FRAMEWORKS = ("langgraph", "strands")
INVOCATIONS = 4                                # separate fresh environments per configuration
REPLICATES = 30                                # x 4 invocations = 120 runs = 4,800 steps
CHUNK = 6                                      # replicates per Lambda call (about 15 s at 512 MB)
LONG = Config(read_timeout=900, connect_timeout=30, retries={"max_attempts": 1, "mode": "standard"})

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
c = lambda svc, cfg=None: session.client(svc, config=cfg) if cfg else session.client(svc)


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=1))


# ---------------------------------------------------------------- check
def check() -> None:
    ident = c("sts").get_caller_identity()
    print(f"profile {PROFILE}, region {REGION}, principal {ident['Arn'].split(':')[-1]}")
    for arch in ARCHS:
        z = ROOT / "cloud" / "build" / f"bench-{arch}.zip"
        print(f"bundle {arch}: {'present' if z.exists() else 'MISSING, run cloud/build_bundle.sh ' + arch}")
    c("lambda").list_functions(MaxItems=1)
    c("bedrock-agentcore-control").list_agent_runtimes(maxResults=1)
    print("lambda and agentcore-control reachable; IAM and S3 permissions are exercised by deploy")


# ---------------------------------------------------------------- deploy
def _role(name: str, service: str, account: str, inline: dict | None, managed: list[str]) -> str:
    iam = c("iam")
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]}
    try:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="agentprof platform CPU benchmark")["Role"]["Arn"]
        print(f"created role {name}")
    except iam.exceptions.EntityAlreadyExistsException:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    for m in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=m)
    if inline:
        iam.put_role_policy(RoleName=name, PolicyName=f"{name}-inline", PolicyDocument=json.dumps(inline))
    return arn


def deploy() -> None:
    s = load_state()
    account = c("sts").get_caller_identity()["Account"]
    s3 = c("s3")
    bucket = s.get("bucket") or f"{PREFIX}-{uuid.uuid4().hex[:12]}"
    try:
        s3.create_bucket(Bucket=bucket)
        print(f"created bucket {bucket}")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou",):
            raise
    s["bucket"] = bucket
    for arch in ARCHS:
        z = ROOT / "cloud" / "build" / f"bench-{arch}.zip"
        s3.upload_file(str(z), bucket, z.name)
        print(f"uploaded {z.name} ({z.stat().st_size / 1e6:.0f} MB)")
    save_state(s)

    s["lambda_role"] = _role(f"{PREFIX}-lambda", "lambda.amazonaws.com", account, None,
                             ["arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"])
    s["agentcore_role"] = _role(f"{PREFIX}-agentcore", "bedrock-agentcore.amazonaws.com", account, {
        "Version": "2012-10-17", "Statement": [
            {"Effect": "Allow",
             "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                        "logs:DescribeLogGroups", "logs:DescribeLogStreams"],
             "Resource": f"arn:aws:logs:{REGION}:{account}:log-group:/aws/bedrock-agentcore/runtimes/*"},
            {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": f"arn:aws:s3:::{bucket}/*"},
        ]}, [])
    save_state(s)
    print("waiting 15 s for IAM propagation"); time.sleep(15)

    lam = c("lambda")
    s.setdefault("functions", {})
    for arch in ARCHS:
        name = f"{PREFIX}-{arch.replace('_', '')}"
        try:
            lam.create_function(FunctionName=name, Runtime="python3.12", Role=s["lambda_role"],
                                Handler="lambda_function.handler",
                                Code={"S3Bucket": bucket, "S3Key": f"bench-{arch}.zip"},
                                Architectures=[arch], MemorySize=1769, Timeout=600,
                                Description="agentprof platform CPU benchmark (no network)")
            print(f"created function {name}")
        except lam.exceptions.ResourceConflictException:
            lam.update_function_code(FunctionName=name, S3Bucket=bucket, S3Key=f"bench-{arch}.zip")
            print(f"updated function {name}")
        lam.get_waiter("function_active_v2").wait(FunctionName=name)
        s["functions"][arch] = name
    save_state(s)

    ctl = c("bedrock-agentcore-control")
    if not s.get("runtime_id"):
        r = ctl.create_agent_runtime(
            agentRuntimeName=RUNTIME_NAME,
            agentRuntimeArtifact={"codeConfiguration": {
                "code": {"s3": {"bucket": bucket, "prefix": "bench-arm64.zip"}},
                "runtime": "PYTHON_3_12", "entryPoint": ["main.py"]}},
            roleArn=s["agentcore_role"], networkConfiguration={"networkMode": "PUBLIC"},
            lifecycleConfiguration={"idleRuntimeSessionTimeout": 120},
            description="agentprof platform CPU benchmark (no network)")
        s["runtime_id"], s["runtime_arn"] = r["agentRuntimeId"], r["agentRuntimeArn"]
        save_state(s)
        print(f"creating AgentCore runtime {RUNTIME_NAME}")
    for _ in range(90):
        st = ctl.get_agent_runtime(agentRuntimeId=s["runtime_id"])
        if st["status"] in ("READY", "CREATE_FAILED", "UPDATE_FAILED"):
            break
        time.sleep(10)
    print(f"AgentCore runtime status: {st['status']} {st.get('failureReason', '')}")
    if st["status"] != "READY":
        sys.exit(1)


# ---------------------------------------------------------------- run
def _save(target: str, fw: str, results: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs, envs = [], []
    for k, r in enumerate(results):
        envs.append(r["environment"])
        for rep in r["runs"]["none"]:
            runs.append({**rep, "invocation": k})
    path = OUT / f"{target}-{fw}.json"
    path.write_text(json.dumps({"target": target, "framework": fw, "environment": envs[0],
                                "environments": envs, "steps": results[0]["steps"],
                                "replicates": len(runs), "runs": {"none": runs},
                                "all_synchronous": all(r["all_synchronous"] for r in results)}, indent=1))
    print(f"  wrote {path.relative_to(ROOT)} ({len(runs)} runs; CPUs seen: "
          f"{sorted({e.get('cpu_brand') for e in envs})})")


def run_lambda(s: dict) -> None:
    lam = c("lambda", LONG)
    for arch in ARCHS:
        name = s["functions"][arch]
        for mem in LAMBDA_MEMORY_MB:
            for fw in FRAMEWORKS:
                results = []
                for k in range(INVOCATIONS):
                    # A configuration change forces a fresh execution environment, so
                    # each group of calls may land on a different host.
                    lam.update_function_configuration(FunctionName=name, MemorySize=mem,
                                                      Environment={"Variables": {"NONCE": uuid.uuid4().hex}})
                    lam.get_waiter("function_updated_v2").wait(FunctionName=name)
                    # The replicates are split into short calls on that same warm
                    # environment: long synchronous calls (about 60 s at 512 MB) were
                    # dropped by the client's network path before the response arrived.
                    t0, merged = time.time(), None
                    for chunk in range(REPLICATES // CHUNK):
                        resp = lam.invoke(FunctionName=name, Payload=json.dumps(
                            {"framework": fw, "replicates": CHUNK, "warmup": 2 if chunk == 0 else 0}).encode())
                        body = json.loads(resp["Payload"].read())
                        if resp.get("FunctionError"):
                            raise RuntimeError(f"{name} {mem} MB {fw}: {body}")
                        if merged is None:
                            merged = body
                        else:
                            if body["environment"]["cpu_brand"] != merged["environment"]["cpu_brand"]:
                                print(f"  note: CPU changed within group: {body['environment']['cpu_brand']}")
                            merged["runs"]["none"] += body["runs"]["none"]
                            merged["all_synchronous"] = merged["all_synchronous"] and body["all_synchronous"]
                    for i, r in enumerate(merged["runs"]["none"]):
                        r["replicate"] = i
                    results.append(merged)
                    print(f"lambda {arch} {mem} MB {fw} #{k}: {time.time() - t0:.0f} s, "
                          f"cpu {merged['environment']['cpu_brand']}")
                _save(f"lambda-{arch}-{mem}", fw, results)


def run_agentcore(s: dict) -> None:
    data = c("bedrock-agentcore", LONG)
    for fw in FRAMEWORKS:
        results = []
        for k in range(INVOCATIONS):
            sid = f"agentprof-bench-{uuid.uuid4().hex}"          # new key: new microVM
            t0 = time.time()
            resp = data.invoke_agent_runtime(agentRuntimeArn=s["runtime_arn"], runtimeSessionId=sid,
                                             contentType="application/json", accept="application/json",
                                             payload=json.dumps({"framework": fw, "replicates": REPLICATES}).encode())
            body = json.loads(resp["response"].read())
            # Close the session explicitly rather than leave it to the idle timeout.
            try:
                data.stop_runtime_session(agentRuntimeArn=s["runtime_arn"], runtimeSessionId=sid)
            except ClientError as e:
                print(f"  stop_runtime_session: {e.response['Error']['Code']}")
            if "error" in body:
                raise RuntimeError(f"agentcore {fw}: {body['error']}")
            results.append(body)
            print(f"agentcore {fw} #{k}: {time.time() - t0:.0f} s, cpu {body['environment']['cpu_brand']}, "
                  f"cgroup cpu.max {body['environment']['cgroup_cpu_max']}, pings {body.get('pings_during_run')}")
        _save("agentcore", fw, results)


def run() -> None:
    s = load_state()
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("all", "agentcore"):
        run_agentcore(s)
    if which in ("all", "lambda"):
        run_lambda(s)


# ---------------------------------------------------------------- teardown
def teardown() -> None:
    s = load_state()
    if s.get("runtime_id"):
        try:
            c("bedrock-agentcore-control").delete_agent_runtime(agentRuntimeId=s["runtime_id"])
            print("deleted AgentCore runtime")
        except ClientError as e:
            print(f"runtime: {e.response['Error']['Code']}")
    for name in (s.get("functions") or {}).values():
        try:
            c("lambda").delete_function(FunctionName=name); print(f"deleted function {name}")
        except ClientError as e:
            print(f"{name}: {e.response['Error']['Code']}")
    iam = c("iam")
    for role in (f"{PREFIX}-lambda", f"{PREFIX}-agentcore"):
        try:
            for p in iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]:
                iam.detach_role_policy(RoleName=role, PolicyArn=p["PolicyArn"])
            for p in iam.list_role_policies(RoleName=role)["PolicyNames"]:
                iam.delete_role_policy(RoleName=role, PolicyName=p)
            iam.delete_role(RoleName=role); print(f"deleted role {role}")
        except ClientError as e:
            print(f"{role}: {e.response['Error']['Code']}")
    if s.get("bucket"):
        b = session.resource("s3").Bucket(s["bucket"])
        try:
            b.objects.all().delete(); b.delete(); print(f"deleted bucket {s['bucket']}")
        except ClientError as e:
            print(f"bucket: {e.response['Error']['Code']}")
    logs = c("logs")
    for prefix in (f"/aws/lambda/{PREFIX}", f"/aws/bedrock-agentcore/runtimes/{RUNTIME_NAME}"):
        for g in logs.describe_log_groups(logGroupNamePrefix=prefix).get("logGroups", []):
            logs.delete_log_group(logGroupName=g["logGroupName"]); print(f"deleted log group {g['logGroupName']}")
    STATE.unlink(missing_ok=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"check": check, "deploy": deploy, "run": run, "teardown": teardown}.get(
        cmd, lambda: sys.exit(__doc__))()
