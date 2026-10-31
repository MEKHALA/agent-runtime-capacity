# Platform CPU benchmark on AWS (Lambda and AgentCore Runtime)

This runs the paper's transport-mock benchmark (LangGraph and Strands, 40
steps) **on the platforms' own vCPUs**. It turns the deployed CPU per step,
currently a bracket built from a measurement and an estimate, into a
measurement. It also adds CPU measurements from the platforms' own processors, rather
than from one development machine.

**No model is called.** The model client's socket is faked, so no Bedrock
access or tokens are needed. You pay only for a few minutes of Lambda and
AgentCore compute.

## What gets created

All resources are prefixed `agentprof-bench` (the runtime is
`agentprof_bench`), and `teardown` deletes all of them.

| Resource | Why |
|---|---|
| S3 bucket `agentprof-bench-<random>` | Holds the two deployment zips (about 50 MB each) |
| IAM role `agentprof-bench-lambda` | Lambda execution: CloudWatch Logs only |
| IAM role `agentprof-bench-agentcore` | AgentCore execution: its own log group, and read access to the zip |
| Lambda `agentprof-bench-arm64`, `agentprof-bench-x8664` | Python 3.12, 600 s timeout; memory varied 512 / 1,024 / 1,769 / 3,008 MB |
| AgentCore runtime `agentprof_bench` | Direct code deployment (arm64 zip), public network mode, 120 s idle timeout |

## The runs

- **Each configuration:** 4 separate fresh environments × 30 replicates ×
  40 steps = 4,800 steps per framework. That matches the laptop and
  desktop sweeps.
- **Lambda:** 2 architectures × 4 memory sizes × 2 frameworks = 16
  configurations. Lambda allocates CPU in proportion to memory: 1,769 MB is
  one vCPU. So the memory sweep shows how framework CPU per step stretches
  when the environment is small.
- **AgentCore:** 2 frameworks, each in 4 new sessions. A new session key
  means a new microVM. Every session is stopped explicitly after its run.
- **Wall time:** about 45–60 minutes in total, mostly Lambda at 512 MB.

**Expected cost: under $2.** About 70 Lambda invocations of 10–90 s at
0.5–3 GB, 8 AgentCore sessions of about a minute, and a few hundred MB of
S3 storage held for an hour.

## Prerequisites

### 1. A profile named `agentprof` on this machine

**Do not paste access keys into the chat.** Create the files yourself in a
terminal on this machine:

```bash
mkdir -p ~/.aws && chmod 700 ~/.aws
cat >> ~/.aws/config <<'EOF'
[profile agentprof]
region = us-east-1
EOF
nano ~/.aws/credentials     # add the block below, with your own keys
chmod 600 ~/.aws/credentials
```

The block to add:

```ini
[agentprof]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
```

Use your personal study account, the same one the AgentCore study used.
Ideally, use a **dedicated IAM user** that holds only the policy below.
**Delete its access key** when the runs are done.

### 2. Permissions for that user

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "Identity", "Effect": "Allow", "Action": "sts:GetCallerIdentity", "Resource": "*"},
    {"Sid": "S3", "Effect": "Allow",
     "Action": ["s3:CreateBucket", "s3:DeleteBucket", "s3:ListBucket",
                "s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
     "Resource": ["arn:aws:s3:::agentprof-bench-*", "arn:aws:s3:::agentprof-bench-*/*"]},
    {"Sid": "IAM", "Effect": "Allow",
     "Action": ["iam:CreateRole", "iam:GetRole", "iam:DeleteRole", "iam:PassRole",
                "iam:AttachRolePolicy", "iam:DetachRolePolicy", "iam:ListAttachedRolePolicies",
                "iam:PutRolePolicy", "iam:DeleteRolePolicy", "iam:ListRolePolicies"],
     "Resource": "arn:aws:iam::*:role/agentprof-bench-*"},
    {"Sid": "Lambda", "Effect": "Allow",
     "Action": ["lambda:CreateFunction", "lambda:UpdateFunctionCode", "lambda:UpdateFunctionConfiguration",
                "lambda:GetFunction", "lambda:GetFunctionConfiguration", "lambda:InvokeFunction",
                "lambda:DeleteFunction", "lambda:ListFunctions"],
     "Resource": "*"},
    {"Sid": "AgentCore", "Effect": "Allow",
     "Action": ["bedrock-agentcore:CreateAgentRuntime", "bedrock-agentcore:GetAgentRuntime",
                "bedrock-agentcore:DeleteAgentRuntime", "bedrock-agentcore:ListAgentRuntimes",
                "bedrock-agentcore:InvokeAgentRuntime", "bedrock-agentcore:StopRuntimeSession",
                "bedrock-agentcore:CreateWorkloadIdentity", "bedrock-agentcore:GetWorkloadIdentity",
                "bedrock-agentcore:DeleteWorkloadIdentity"],
     "Resource": "*"},
    {"Sid": "Logs", "Effect": "Allow",
     "Action": ["logs:DescribeLogGroups", "logs:DeleteLogGroup", "logs:FilterLogEvents", "logs:GetLogEvents"],
     "Resource": "*"}
  ]
}
```

If `deploy` fails with an AccessDenied error that names an action not in
this list, add that action and re-run. `deploy` is safe to re-run. The
workload-identity actions are my best reading of what CreateAgentRuntime
needs; they may be unnecessary.

## Steps

```bash
cd ~/src/agentprof
cloud/build_bundle.sh arm64 && cloud/build_bundle.sh x86_64   # already built once; re-run after code changes
.venv/bin/python cloud/aws.py check       # reads credentials, confirms the services are reachable
.venv/bin/python cloud/aws.py deploy      # about 3-5 minutes; most of it is AgentCore runtime creation
.venv/bin/python cloud/aws.py run agentcore   # about 10 minutes; run this first, it matters most
.venv/bin/python cloud/aws.py run lambda      # about 40 minutes
.venv/bin/python cloud/aws.py teardown    # deletes everything above, plus its log groups
```

Results land in `data/platform/<target>-<framework>.json`, in the same
schema as the other raw files, so `scripts/sensitivity.py` can fit them
directly.

## Things that could go wrong on the first deploy

- **AgentCore `CREATE_FAILED`.** The runtime scans native binaries for
  arm64. The build script already checks this locally, but the failure
  reason will say which file. The other likely cause: the S3 `prefix`
  field is the object *key* (`bench-arm64.zip`). The API docs call it a
  prefix, and the examples use the full key.
- **AgentCore health check.** `main.py` serves `/ping` and `/invocations`
  on port 8080, using the standard library only. It answers pings during a
  run on a separate thread, and records how many arrived, because
  Strands' process-CPU clock would include them.
- **The Lambda bundle has not been executed on Python 3.12 yet.** This
  machine has only Python 3.14, so the bundle was smoke-tested only for
  import completeness. The first `run` will show any problem within
  seconds.
