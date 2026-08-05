"""Staging a cell and running it for a while.

A cell runs as three containers on two networks. The agent and the evaluator
share a network that reaches nothing else, so the only way out of the cell is
the proxy, which joins both and forwards to the model providers alone. The
agent holds the credential; the evaluator holds the scoring code and the score
file, and the agent can reach it only over HTTP. Nothing the agent writes can
change what it is scored on.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import docker
from docker.errors import APIError, ImageNotFound

from stupidbench.cell import (
    BUDGET_SECONDS,
    FLOWS,
    HOME_VARIABLES,
    STATE_DIRS,
    TOKEN_VARIABLES,
    Cell,
    Tool,
)

TASK_DIR = Path(__file__).resolve().parent / "task"

#: A cell runs in a flowbench runtime image, by the variant its task calls for
#: — `latest` is the one aopt is built and scored against. The package is
#: public, so pulling it needs no credential. Both parts are overridable, so a
#: task that needs another variant, or a test that has a local build, can say.
RUNTIME_REPOSITORY = "ghcr.io/humanfia/flowbench-runtime"
RUNTIME_VARIANT = os.environ.get("STUPIDBENCH_VARIANT", "latest")
IMAGE = os.environ.get("STUPIDBENCH_IMAGE", f"{RUNTIME_REPOSITORY}:{RUNTIME_VARIANT}")
PROXY_IMAGE = (
    "docker.io/3proxy/3proxy:0.9.7.busybox"
    "@sha256:8b38b23ab45e1b038e620c2e507994af801df2ca7dc78f9279367b1c89f68cd8"
)

EVALUATOR_HOST = "evaluator"
PROXY_HOST = "proxy"
PROXY_PORT = 8888
RESOLVER_PORT = 8889

#: Where a model provider is reached, and nothing else is. A cell that could
#: open a connection anywhere could post the credential it holds anywhere.
ALLOWED_HOSTS = (
    "api.anthropic.com",
    "platform.claude.com",
    "chatgpt.com",
    "auth.openai.com",
    "api.kimi.com",
    "api.moonshot.ai",
)

AGENT_MEMORY_MB = 6144
AGENT_CPU = 2
EVALUATOR_MEMORY_MB = 512
EVALUATOR_CPU = 1
PROXY_MEMORY_MB = 64

READINESS_SECONDS = 90
TICK_SECONDS = 30

#: Kimi Code reads an ad-hoc model out of its environment and never writes it to
#: disk; the managed provider it would otherwise use needs an interactive login
#: that a runner cannot give it. The default capabilities leave out tool use,
#: which an agent cannot work without.
#: Where the key is redeemed. A coding-plan key and a platform key are taken
#: at different endpoints, and which one a runner holds is not knowable from
#: here, so a repository variable can say.
KIMI_BASE_URL = (
    os.environ.get("KIMI_MODEL_BASE_URL") or "https://api.kimi.com/coding/v1"
)

_KIMI_ENVIRONMENT = {
    "KIMI_MODEL_PROVIDER_TYPE": "kimi",
    "KIMI_MODEL_MAX_CONTEXT_SIZE": "1048576",
    "KIMI_MODEL_CAPABILITIES": "thinking,always_thinking,image_in,tool_use",
}

_RUN_SCRIPTS: dict[Tool, str] = {
    "codex": """codex exec \\
        --dangerously-bypass-approvals-and-sandbox \\
        --skip-git-repo-check \\
        --model "$MODEL" \\
        -c "model_reasoning_effort=\\"$EFFORT\\"" \\
        -c 'service_tier="default"' \\
        < .stupidbench/task.md || true""",
    # Claude keeps a copy of the settings its provider hands down, and one of
    # them turns off the very mode the cell runs in: a run that starts with that
    # copy on disk asks for an approval nobody is there to give, and every
    # command it tries comes back refused. The copy is a cache, fetched again
    # each time, so a run starts without one.
    "claude": """rm -f "$CLAUDE_CONFIG_DIR/remote-settings.json"
    claude --print \\
        --dangerously-skip-permissions \\
        --model "$MODEL" \\
        --effort "$EFFORT" \\
        < .stupidbench/task.md || true""",
    # The env-configured model is the default one, so naming it would name the
    # alias rather than the model, and the effort travels in the environment.
    "kimi": """kimi --prompt "$(cat .stupidbench/task.md)" || true""",
}


def prepare(cell: Cell) -> None:
    """Stages the cell, or leaves alone one a previous segment already staged.

    The run script is the exception: it is written every segment. It is the one
    file here that is ours rather than the agent's, and a cell that was already
    in flight when a bug in it was found has the rest of its budget to spend on
    the fixed one.
    """
    flow = FLOWS[cell.flow]
    agent = cell.agent_dir
    (agent / ".stupidbench").mkdir(parents=True, exist_ok=True)
    (agent / ".stupidbench/run.sh").write_text(
        "#!/bin/bash\n\n"
        f'export MODEL="{flow.model}"\n'
        f'export EFFORT="{flow.effort}"\n\n'
        "while true; do\n"
        f"    {_RUN_SCRIPTS[flow.tool]}\n"
        "    sleep 5\n"
        "done\n",
        encoding="utf-8",
    )
    if cell.events_path.is_file():
        return
    (agent / "tests").mkdir(parents=True, exist_ok=True)
    for source, destination in (
        ("TASK.md", ".stupidbench/task.md"),
        ("perf_takehome.py", "perf_takehome.py"),
        ("problem.py", "problem.py"),
        ("tests/generate.py", "tests/generate.py"),
        ("tests/submission_tests.py", "tests/submission_tests.py"),
    ):
        shutil.copy2(TASK_DIR / source, agent / destination)
    # A CLI's web tools run at its provider, so they reach hosts the cell's own
    # network never can. Every flow denies them, so that what an agent finds is
    # what it worked out. Kimi Code needs no entry: its search is a service the
    # ad-hoc model is never given.
    state_dir = agent / STATE_DIRS[flow.tool]
    state_dir.mkdir(parents=True, exist_ok=True)
    if flow.tool == "claude":
        (state_dir / "settings.json").write_text(
            '{\n  "permissions": {\n    "deny": ["WebSearch", "WebFetch"]\n  }\n}\n',
            encoding="utf-8",
        )
    if flow.tool == "codex":
        (state_dir / "config.toml").write_text(
            'web_search = "disabled"\n\n[tools]\nweb_search = false\n',
            encoding="utf-8",
        )

    evaluator = cell.evaluator_dir
    evaluator.mkdir(parents=True, exist_ok=True)
    for name in ("evaluator_server.py", "frozen_problem.py", "score.py"):
        shutil.copy2(TASK_DIR / name, evaluator / name)

    cell.scores_path.touch()
    cell.events_path.touch()


def run(cell: Cell, seconds: int) -> str:
    """Runs the cell for ``seconds``, or for what is left of its budget.

    Returns what became of it: "done" when the budget is spent, "ran" when the
    segment carried it further.
    """
    if not (cell.agent_dir / ".stupidbench/run.sh").is_file():
        # Docker would make the missing mount point itself, owned by root, and
        # the cell would be unusable from then on.
        raise RuntimeError(f"{cell.flow}/{cell.seed} has not been staged")
    remaining = BUDGET_SECONDS - cell.elapsed
    if remaining <= 0:
        return "done"
    limit = int(min(seconds, remaining))
    flow = FLOWS[cell.flow]
    client = docker.from_env()
    for image in (PROXY_IMAGE, IMAGE):
        try:
            client.images.get(image)
        except ImageNotFound:
            client.images.pull(image)

    proxy_dir = Path(tempfile.mkdtemp(prefix="stupidbench-proxy-"))
    resolver_path = proxy_dir / "resolver.cfg"
    resolver_path.write_text(_resolver_config(), encoding="utf-8")
    resolver_path.chmod(0o444)
    # Written once the resolver has an address, which is the one thing in the
    # gate's config that only exists after a container does.
    config_path = proxy_dir / "3proxy.cfg"
    name = f"stupidbench-{cell.flow}-{cell.seed}"
    # A segment that died without cleaning up leaves containers holding the
    # names this one needs, and creating them again would fail. The cell's own
    # names are its to reclaim.
    for container in client.containers.list(all=True, filters={"name": name}):
        container.remove(force=True, v=True)
    for network in client.networks.list(names=[f"{name}-internal", f"{name}-egress"]):
        network.remove()
    containers = []
    networks = []
    # Only a start can be stopped. A segment that fails while setting up has
    # run for no time at all, and an unpaired stop would charge the cell for
    # every hour since the segment before it.
    started = False
    try:
        internal = client.networks.create(
            f"{name}-internal",
            driver="bridge",
            internal=True,
            options={"com.docker.network.bridge.gateway_mode_ipv4": "isolated"},
        )
        networks.append(internal)
        egress = client.networks.create(f"{name}-egress", driver="bridge")
        networks.append(egress)

        evaluator = client.containers.create(
            IMAGE,
            command=["python3", "evaluator_server.py"],
            name=f"{name}-evaluator",
            hostname=EVALUATOR_HOST,
            working_dir="/workspace",
            environment={"HOME": "/workspace"},
            user=f"{os.getuid()}:{os.getgid()}",
            mem_limit=f"{EVALUATOR_MEMORY_MB}m",
            nano_cpus=EVALUATOR_CPU * 1_000_000_000,
            volumes={
                str(cell.evaluator_dir.resolve()): {"bind": "/workspace", "mode": "rw"},
                str(cell.scores_path.resolve()): {
                    "bind": "/workspace/scores.jsonl",
                    "mode": "rw",
                },
            },
            network=internal.name,
            dns=["127.0.0.1"],
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            use_config_proxy=False,
        )
        containers.append(evaluator)
        # The only thing in a cell that ever resolves a name, and the agent is
        # on neither of the networks it can be reached from.
        resolver = client.containers.create(
            PROXY_IMAGE,
            name=f"{name}-resolver",
            hostname="resolver",
            user="65534:65534",
            mem_limit=f"{PROXY_MEMORY_MB}m",
            volumes={
                str(resolver_path): {"bind": "/etc/3proxy/3proxy.cfg", "mode": "ro"}
            },
            network=egress.name,
            read_only=True,
            tmpfs={"/tmp": "rw,noexec,nosuid,size=16m"},
            sysctls={"net.ipv4.ip_forward": "0"},
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            use_config_proxy=False,
        )
        containers.append(resolver)
        resolver.start()
        config_path.write_text(
            _proxy_config(_address(resolver, egress.name)), encoding="utf-8"
        )
        config_path.chmod(0o444)
        proxy = client.containers.create(
            PROXY_IMAGE,
            name=f"{name}-proxy",
            hostname=PROXY_HOST,
            user="65534:65534",
            mem_limit=f"{PROXY_MEMORY_MB}m",
            volumes={
                str(config_path): {"bind": "/etc/3proxy/3proxy.cfg", "mode": "ro"}
            },
            network=egress.name,
            # The resolver it forwards to is reached by address, so the one
            # Docker would otherwise hand it is a way out it has no use for.
            dns=["127.0.0.1"],
            read_only=True,
            tmpfs={"/tmp": "rw,noexec,nosuid,size=16m"},
            sysctls={"net.ipv4.ip_forward": "0"},
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            use_config_proxy=False,
        )
        containers.append(proxy)
        internal.connect(proxy, aliases=[PROXY_HOST])
        evaluator.start()
        proxy.start()
        _await_evaluator(evaluator)
        # A resolver that did not stay up leaves the gate forwarding to an
        # address nobody answers, and the cell would spend the rest of its
        # segment unable to reach a provider without anything saying why.
        # Docker still calls a container that has just died running, so this is
        # asked once the evaluator is up rather than when the resolver started.
        resolver.reload()
        if resolver.status != "running":
            raise RuntimeError("the cell's resolver did not stay up")

        agent = client.containers.create(
            IMAGE,
            command=["timeout", str(limit), "bash", ".stupidbench/run.sh"],
            name=f"{name}-agent",
            hostname="agent",
            working_dir="/workspace",
            environment=_agent_environment(flow.tool, flow.model, flow.effort),
            user=f"{os.getuid()}:{os.getgid()}",
            mem_limit=f"{AGENT_MEMORY_MB}m",
            nano_cpus=AGENT_CPU * 1_000_000_000,
            volumes={
                str(cell.agent_dir.resolve()): {"bind": "/workspace", "mode": "rw"}
            },
            network=internal.name,
            dns=["127.0.0.1"],
            extra_hosts={
                EVALUATOR_HOST: _address(evaluator, internal.name),
                PROXY_HOST: _address(proxy, internal.name),
            },
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            use_config_proxy=False,
        )
        containers.append(agent)
        agent.start()
        cell.record("start")
        started = True
        _watch(cell, agent, limit)
        (cell.cell_dir / "agent.log").write_bytes(agent.logs(tail=400))
    finally:
        if started:
            cell.record("stop")
        for container in reversed(containers):
            try:
                container.remove(force=True, v=True)
            except APIError:
                pass
        for network in networks:
            try:
                network.remove()
            except APIError:
                pass
        shutil.rmtree(proxy_dir, ignore_errors=True)
    return "ran"


def _watch(cell: Cell, agent, limit: int) -> None:
    """Ticks while the agent runs, so a segment that dies still bounds itself."""
    deadline = time.monotonic() + limit + TICK_SECONDS
    while time.monotonic() < deadline:
        time.sleep(TICK_SECONDS)
        agent.reload()
        if agent.status != "running":
            return
        cell.record("tick")


def _await_evaluator(evaluator) -> None:
    deadline = time.monotonic() + READINESS_SECONDS
    probe = [
        "python3",
        "-c",
        "import urllib.request;urllib.request.urlopen('http://127.0.0.1:80/scores')",
    ]
    while time.monotonic() < deadline:
        evaluator.reload()
        if evaluator.status != "running":
            raise RuntimeError("evaluator exited before it was ready")
        if evaluator.exec_run(probe).exit_code == 0:
            return
        time.sleep(2)
    raise RuntimeError("evaluator never became ready")


def _agent_environment(tool: Tool, model: str, effort: str) -> dict[str, str]:
    proxy_url = f"http://{PROXY_HOST}:{PROXY_PORT}"
    no_proxy = f"localhost,127.0.0.1,::1,{EVALUATOR_HOST}"
    environment = {
        "HOME": "/workspace",
        HOME_VARIABLES[tool]: f"/workspace/{STATE_DIRS[tool]}",
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "NO_PROXY": no_proxy,
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
        "no_proxy": no_proxy,
    }
    token = os.environ.get(TOKEN_VARIABLES[tool])
    if token:
        environment[TOKEN_VARIABLES[tool]] = token
    if tool == "kimi":
        environment |= _KIMI_ENVIRONMENT | {
            "KIMI_MODEL_BASE_URL": KIMI_BASE_URL,
            "KIMI_MODEL_NAME": model.rpartition("/")[2],
            "KIMI_MODEL_THINKING_EFFORT": effort,
        }
    return environment


def _address(container, network: str) -> str:
    container.reload()
    address = container.attrs["NetworkSettings"]["Networks"][network]["IPAddress"]
    if container.status != "running" or not address:
        raise RuntimeError(f"{container.name} is not on {network}")
    return address


def _proxy_config(resolver: str) -> str:
    """The half an agent talks to, which never resolves a name.

    3proxy looks a target up before it applies its rules, so a cell that only
    denied the connection would already have sent ``<secret>.attacker.example``
    to a nameserver the attacker owns: the answer is refused, the question is
    the exfiltration. Under ``fakeresolve`` nothing is looked up here at all — a
    host is allowed on its name alone, and only a name that was allowed is
    handed on, still a name, to the half that does resolve.
    """
    return "\n".join(
        (
            "nscache 65536",
            "fakeresolve",
            "log",
            "auth iponly",
            f"allow * * {','.join(ALLOWED_HOSTS)} 443 HTTPS",
            f"parent 1000 http {resolver} {RESOLVER_PORT}",
            "deny *",
            f"proxy -a -p{PROXY_PORT}",
            "",
        )
    )


def _resolver_config() -> str:
    """The half that resolves, on a network the agent is not on.

    It keeps the same allow list rather than trusting what reached it, for the
    reason the evaluator is its own container: one rule is an assumption, two
    that have to agree is a check.
    """
    return "\n".join(
        (
            "nserver 127.0.0.11",
            "nscache 65536",
            "log",
            "auth iponly",
            f"allow * * {','.join(ALLOWED_HOSTS)} 443 HTTPS",
            "deny *",
            f"proxy -a -p{RESOLVER_PORT}",
            "",
        )
    )
