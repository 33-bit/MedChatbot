# Server Setup Manual

This guide covers running the chatbot server with Docker Compose on a local
machine and on a VPS or EC2 instance.

The current Compose stack starts:

- `redis`: local session and quota store
- `qdrant`: local vector database for dense retrieval
- `neo4j`: local graph database used by KG retrieval
- `api`: FastAPI chatbot server built from `Dockerfile`
- `ngrok`: public HTTPS tunnel to the API
- `telegram-webhook`: one-shot container that registers Telegram `setWebhook`
- `zalo-webhook`: one-shot container that registers Zalo Bot Platform `setWebhook`

Secrets must stay in `.env`. Do not put tokens or API keys in `Dockerfile`,
`docker-compose.yml`, or git.

## Required Files

The Docker setup uses these files:

```text
Dockerfile
docker-compose.yml
.dockerignore
.env
.env.docker.example
```

Create `.env` from the example:

```bash
cp .env.docker.example .env
```

Fill the real values in `.env`:

```env
CHAT_API_KEY=...

LLM_API_KEY=...
BASE_URL=...
MODEL=gpt-4.1
FAST_MODEL=gpt-4.1-mini
GUARDRAIL_MODEL=gpt-4.1-mini

QDRANT_URL=...
QDRANT_API_KEY=
QDRANT_PORT=6334
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_HTTP_PORT=7475
NEO4J_BOLT_PORT=7688
REDIS_URL=redis://redis:6379/0

TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET=...
TELEGRAM_DROP_PENDING_UPDATES=true
ZALO_BOT_TOKEN=...
ZALO_WEBHOOK_SECRET=...

NGROK_AUTHTOKEN=...
NGROK_URL=https://your-static-domain.ngrok-free.dev

WEBHOOK_REGISTRATION_DELAY_SECONDS=15
HF_PRELOAD_RETRIEVAL_MODELS=1
HF_PRELOAD_REQUIRED=0
```

The Compose file starts Redis, Qdrant, and Neo4j locally. Inside Docker, the
API connects to Redis at `redis://redis:6379/0`, Qdrant at
`http://qdrant:6333`, and Neo4j at `bolt://neo4j:7687`. From your host
machine, Qdrant is exposed at `http://localhost:6334` and the Neo4j browser is
exposed at `http://localhost:7475` by default.

## Local Setup

### 1. Start Docker

On macOS or Windows, open Docker Desktop first.

Check Docker:

```bash
docker info
docker compose version
```

If you see an error like this:

```text
failed to connect to the docker API
connect: no such file or directory
```

Docker Desktop or the Docker daemon is not running.

On macOS, start it from terminal:

```bash
open -a Docker
```

Wait until Docker Desktop says Docker is running.

### 2. Start the Server

From the project directory:

```bash
docker compose up --build
```

This runs in the foreground and shows logs. Stop with `Ctrl + C`.

To run in the background:

```bash
docker compose up -d --build
```

Check status:

```bash
docker compose ps
```

The API is ready when the `api` service is `healthy`.

Test local health:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"ok":true}
```

Build the local Neo4j knowledge graph and Qdrant collections once after the
first startup:

```bash
docker compose run --rm api python -m src.rag.kg_builder --clear
docker compose run --rm api python -m src.rag.build_qdrant
docker compose run --rm api python -m src.processing.health_insurance.parse
docker compose run --rm api python -m src.rag.chunker --source health_insurance
docker compose run --rm api python -m src.rag.build_qdrant --health-insurance --reset
```

Redis, Qdrant, and Neo4j data persist in the `redis-data`, `qdrant-data`, and
`neo4j-data` Docker volumes. Do not run the KG builder with `--clear` again
unless you want to rebuild the graph.

You do not need to run those two build commands every time you start the stack.
Run them again only when:

- this is the first startup with empty Docker volumes
- you changed the processed entities, chunks, or source data and want updated indexes
- you intentionally want to rebuild the stored graph or vector collections
- you deleted volumes with `docker compose down -v`

Normal `docker compose stop`, `docker compose start`, `docker compose down`,
and `docker compose up -d` keep the named volumes, so the Neo4j graph and
Qdrant collections remain available.

Test public ngrok health:

```bash
curl "$NGROK_URL/health"
```

### 3. Check Telegram Webhook

The `telegram-webhook` service exits after registering the webhook. That is
normal.

Check its logs:

```bash
docker compose logs telegram-webhook
```

Expected log:

```text
Telegram webhook registered: https://your-static-domain.ngrok-free.dev/webhook/telegram
```

If you change `NGROK_URL`, `TELEGRAM_BOT_TOKEN`, or
`TELEGRAM_WEBHOOK_SECRET`, rerun webhook registration:

```bash
docker compose up --force-recreate telegram-webhook
```

### 4. Check Zalo Webhook

The `zalo-webhook` service exits after registering the webhook. That is
normal.

Check its logs:

```bash
docker compose logs zalo-webhook
```

Expected log:

```text
Zalo webhook registered: https://your-static-domain.ngrok-free.dev/webhook/zalo
```

If you change `NGROK_URL`, `ZALO_BOT_TOKEN`, or `ZALO_WEBHOOK_SECRET`, rerun
webhook registration:

```bash
docker compose up --force-recreate zalo-webhook
```

### 4. Stop and Restart

Stop containers but keep them:

```bash
docker compose stop
```

Start stopped containers:

```bash
docker compose start
```

Stop and remove containers:

```bash
docker compose down
```

Start again:

```bash
docker compose up -d
```

Rebuild after code or dependency changes:

```bash
docker compose up -d --build
```

View logs:

```bash
docker compose logs -f api
docker compose logs -f ngrok
docker compose logs telegram-webhook
```

## Podman Setup

Use this section when the host supports Podman but you want the same container
stack without Docker.

### 1. Install Podman

On Ubuntu 20.10 or newer:

```bash
sudo apt-get update
sudo apt-get -y install podman
```

Check Podman:

```bash
podman --version
podman info
```

Podman itself is enough for normal `podman` commands. The `podman compose`
command also needs a Compose provider such as `docker-compose` or
`podman-compose` available on the host.

Check Compose support:

```bash
podman compose version
```

If `podman compose version` says no Compose provider is available, install one
before continuing. The exact package name depends on the Linux distribution.

### 2. Start With Podman Compose

Use:

```bash
podman compose -f podman-compose.yml up -d --build
```

Check status:

```bash
podman compose -f podman-compose.yml ps
```

The service names, environment values, host ports, and named volumes match the
Docker Compose setup above.

### 3. Build Data Once

Build the local Neo4j knowledge graph and Qdrant collections once after the
first startup:

```bash
podman compose -f podman-compose.yml run --rm api python -m src.rag.kg_builder --clear
podman compose -f podman-compose.yml run --rm api python -m src.rag.build_qdrant
```

You do not need to rerun those two build commands on every startup. Run them
again only for empty volumes, changed source data, intentional rebuilds, or if
you removed volumes.

### 4. Stop And Restart

```bash
podman compose -f podman-compose.yml stop
podman compose -f podman-compose.yml start
podman compose -f podman-compose.yml down
podman compose -f podman-compose.yml logs -f api
```

## VPS or EC2 Setup

These steps assume Ubuntu on a VPS or AWS EC2.

Recommended minimum:

```text
CPU: 2 vCPU
RAM: 4 GB minimum, 8 GB safer
Disk: 30 GB minimum, 50 GB safer
Inbound firewall: SSH 22 only if using ngrok
```

Because ngrok provides the public HTTPS URL, you do not need to open port
`8000`, `80`, or `443`.

### 1. SSH Into the Server

For EC2:

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_SERVER_IP
```

For a normal VPS:

```bash
ssh ubuntu@YOUR_SERVER_IP
```

### 2. Install Docker

Choose one container runtime for the VPS:

- use Docker if the host supports a normal Docker daemon
- use Podman if Docker cannot run on that VPS

If your prompt starts with `root@...`, you are already root. Some fresh VPS
images do not include `sudo`. In that case, use the root version below.

Run this on the server:

```bash
sudo rm -f /etc/apt/sources.list.d/docker.list
sudo rm -f /etc/apt/keyrings/docker.asc

sudo apt-get update
sudo apt-get install -y ca-certificates curl git

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

. /etc/os-release

printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' \
  "$(dpkg --print-architecture)" "$VERSION_CODENAME" | \
  sudo tee /etc/apt/sources.list.d/docker.list

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out and SSH back in so the Docker group change applies:

```bash
exit
```

Reconnect, then check:

```bash
docker --version
docker compose version
docker ps
```

If Docker's APT repo does not support your Ubuntu codename, use Ubuntu 22.04
LTS or Ubuntu 24.04 LTS for the server.

Root-only fresh VPS version, without `sudo`:

```bash
rm -f /etc/apt/sources.list.d/docker.list
rm -f /etc/apt/keyrings/docker.asc

apt-get update
apt-get install -y ca-certificates curl git

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

. /etc/os-release

printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' \
  "$(dpkg --print-architecture)" "$VERSION_CODENAME" | \
  tee /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker
```

Check Docker:

```bash
docker --version
docker compose version
docker ps
```

When running as `root`, skip:

```bash
usermod -aG docker "$USER"
```

### 3. Install Podman Instead

Use this path when you want Podman on the VPS instead of Docker.

Normal Ubuntu user with `sudo`:

```bash
sudo apt-get update
sudo apt-get -y install podman
```

Check Podman:

```bash
podman --version
podman info
```

Podman itself is enough for normal `podman` commands. The `podman compose`
command also needs a Compose provider such as `docker-compose` or
`podman-compose` available on the host.

Check Compose support:

```bash
podman compose version
```

If your prompt starts with `root@...`, use the same commands without `sudo`:

```bash
apt-get update
apt-get -y install podman
podman --version
podman info
podman compose version
```

If `podman compose version` says no Compose provider is available, install one
before continuing. The exact package name depends on the Linux distribution.

### 4. Put the Project on the Server

Option A, clone from git:

```bash
git clone <your-repo-url> Chatbot
cd Chatbot
```

Option B, copy from your local machine with `rsync`.

Install `rsync` locally if needed.

On macOS:

```bash
brew install rsync
```

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y rsync
```

Install `rsync` on the server too. If the server prompt is `root@...`:

```bash
apt-get update
apt-get install -y rsync
```

If the server uses a normal sudo user:

```bash
sudo apt-get update
sudo apt-get install -y rsync
```

Copy the project with password SSH:

```bash
rsync -avz \
  --exclude ".git" \
  --exclude ".env" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude "outputs" \
  --exclude "documents" \
  -e "ssh -p YOUR_SSH_PORT" \
  ./ root@YOUR_SERVER_HOST:/root/Chatbot/
```

Copy the project with an SSH key:

```bash
chmod 400 your-key.pem

rsync -avz \
  --exclude ".git" \
  --exclude ".env" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude "outputs" \
  --exclude "documents" \
  -e "ssh -i your-key.pem -p YOUR_SSH_PORT" \
  ./ root@YOUR_SERVER_HOST:/root/Chatbot/
```

Copy `.env` separately if needed:

```bash
scp -P YOUR_SSH_PORT .env root@YOUR_SERVER_HOST:/root/Chatbot/.env
```

With an SSH key:

```bash
scp -i your-key.pem -P YOUR_SSH_PORT .env root@YOUR_SERVER_HOST:/root/Chatbot/.env
```

Fallback with `scp` if you do not want to install `rsync`:

```bash
scp -P YOUR_SSH_PORT -r ./Chatbot root@YOUR_SERVER_HOST:/root/Chatbot
```

With an SSH key:

```bash
scp -i your-key.pem -P YOUR_SSH_PORT -r ./Chatbot root@YOUR_SERVER_HOST:/root/Chatbot
```

`ssh` uses lowercase `-p` for the port. `scp` uses uppercase `-P`.

Do not copy private key files into the repo. Do not commit `.env`.

### 5. Configure `.env`

On the server:

```bash
cp .env.docker.example .env
nano .env
```

Fill the same values used locally.

Save in nano:

```text
Ctrl + O
Enter
Ctrl + X
```

If Redis, Qdrant, or Neo4j are external cloud services, replace the local
service URLs and remove the matching Compose service if you no longer want the
local container.

If they run on the same VPS outside Docker, do not use `localhost` from inside
the container unless the service is also in the Compose network. Prefer a real
network address, or add those services to Compose.

### 6. Start on VPS or EC2

From the project directory:

```bash
docker compose up -d --build
```

Check:

```bash
docker compose ps
docker compose logs -f api
```

Health checks:

```bash
curl http://localhost:8000/health
curl "$NGROK_URL/health"
```

### 7. Restart After Server Reboot

The `api` and `ngrok` services use:

```yaml
restart: unless-stopped
```

After a VPS or EC2 stop/start, Docker should restart them automatically if
Docker is enabled:

```bash
sudo systemctl enable docker
```

After reboot, check:

```bash
cd ~/Chatbot
docker compose ps
```

If they are not running:

```bash
docker compose up -d
```

The `telegram-webhook` container is one-shot and remains exited after success.
That is normal. Rerun it only if the public URL or Telegram secret changes.

### 8. Deploy Code Changes

If the project is cloned from git:

```bash
cd ~/Chatbot
git pull
docker compose up -d --build
```

If you changed only `.env`:

```bash
docker compose up -d
```

If you changed source code, `requirements.txt`, or `Dockerfile`:

```bash
docker compose up -d --build
```

## Data and Persistence

SQLite is stored on the host machine in:

```text
./outputs/chatbot.db
```

Inside the API container, the same file appears at:

```text
/app/outputs/chatbot.db
```

This is configured by:

```yaml
volumes:
  - ./outputs:/app/outputs
environment:
  SQLITE_PATH: /app/outputs/chatbot.db
```

The Hugging Face cache is stored in the Docker named volume:

```text
hf-cache
```

This avoids downloading retrieval models again every time the container is
recreated.

## Bare-Metal Install on Ubuntu

Use this when you cannot or do not want to run Docker on the Ubuntu host
(for example, a bare EC2 instance without Compose, or a local VM). Each
service is installed natively and managed by `systemd`. The API itself can
still run from `python run.py` or from a `systemd` unit.

All commands assume Ubuntu 22.04 or 24.04 and `sudo` access.

### Qdrant

The easiest path is the official Docker image. If Docker is unavailable,
use the prebuilt binary.

```bash
# Option A: Docker
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Option B: Bare-metal binary
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz \
  -o qdrant.tgz
tar xzf qdrant.tgz
sudo mv qdrant /usr/local/bin/
qdrant --version
```

The dashboard lives at `http://localhost:6333/dashboard`. Configure auth by
setting `QDRANT__SERVICE__API_KEY` in `/etc/qdrant/config.yaml` or by
pointing the client at an API key. For the chatbot, set:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

### Redis

```bash
sudo apt update
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
redis-cli ping   # expect: PONG
```

To enable a password, edit `/etc/redis/redis.conf`:

```conf
requirepass yourpassword
```

Then restart and verify:

```bash
sudo systemctl restart redis-server
redis-cli -a yourpassword ping
```

Point the chatbot at it with:

```env
REDIS_URL=redis://:yourpassword@localhost:6379/0
```

### Neo4j

Install Neo4j 5 from the official Debian repo. The package pulls in a
suitable OpenJDK as a dependency.

```bash
wget -O - https://debian.neo4j.com/neotechnology.gpg.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/neo4j.gpg

echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable 5" \
  | sudo tee /etc/apt/sources.list.d/neo4j.list

sudo apt update
sudo apt install -y neo4j
sudo systemctl enable --now neo4j
sudo neo4j-admin dbms set-initial-password yourpassword
```

The browser UI is at `http://localhost:7474`. Configure the chatbot with:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yourpassword
```

### Docker Compose alternative

If you only need a quick local stack on a machine that does have Docker
installed, a single `docker-compose.yml` covers all three services:

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_storage:/qdrant/storage]
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass changeme
    ports: ["6379:6379"]
  neo4j:
    image: neo4j:5
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/changeme
    volumes: [neo4j_data:/data]
volumes:
  qdrant_storage:
  neo4j_data:
```

```bash
docker compose up -d
```

### Verify everything

```bash
curl -s http://localhost:6333/collections | jq .       # Qdrant
redis-cli -a yourpassword ping                         # Redis
cypher-shell -u neo4j -p yourpassword "RETURN 1;"      # Neo4j
```

### Notes

- Qdrant and Redis are required for the chatbot. Neo4j is optional and
  used only by the KG retrieval path; the app starts without it as long
  as `NEO4J_URI` is unset or unreachable.
- Never expose ports `6333`, `6379`, or `7474` to the public internet.
  Put them behind a reverse proxy with TLS if remote access is needed.

## Troubleshooting

### Docker daemon is not running locally

Error:

```text
failed to connect to the docker API
```

Fix:

```bash
open -a Docker
docker info
```

Then rerun:

```bash
docker compose up --build
```

### APT says Docker source list is malformed

Error:

```text
Malformed entry 1 in list file /etc/apt/sources.list.d/docker.list
```

Fix:

```bash
sudo rm -f /etc/apt/sources.list.d/docker.list

. /etc/os-release

printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' \
  "$(dpkg --print-architecture)" "$VERSION_CODENAME" | \
  sudo tee /etc/apt/sources.list.d/docker.list

cat /etc/apt/sources.list.d/docker.list
sudo apt-get update
```

The file must contain one line only.

### Docker build says no space left on device

Error:

```text
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

Check disk:

```bash
df -h
docker system df
```

Clean failed builds:

```bash
docker builder prune -af
docker system prune -af
```

Increase the VPS or EC2 disk to at least 30 GB, preferably 50 GB.

On common Ubuntu EC2 with `/dev/nvme0n1p1`:

```bash
lsblk
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h
```

Then rebuild:

```bash
docker compose up -d --build
```

### API stays in health starting

Check logs:

```bash
docker compose logs -f api
```

The first startup may download and load Hugging Face models. Wait until you see:

```text
Application startup complete.
```

If startup is too slow and you want the API to become ready faster, set this in
`.env`:

```env
HF_PRELOAD_RETRIEVAL_MODELS=0
```

Then restart:

```bash
docker compose up -d --build
```

### Telegram webhook failed

Check:

```bash
docker compose logs telegram-webhook
```

Common causes:

- `TELEGRAM_BOT_TOKEN` is missing or wrong
- `TELEGRAM_WEBHOOK_SECRET` is missing or different from the app value
- `NGROK_URL` is wrong or has a trailing slash problem
- ngrok has not started yet

Rerun:

```bash
docker compose up --force-recreate telegram-webhook
```

### ngrok failed

Check:

```bash
docker compose logs ngrok
```

Common causes:

- `NGROK_AUTHTOKEN` is missing or wrong
- `NGROK_URL` is not assigned to your ngrok account
- you are using a random ngrok URL and Telegram still points to the old URL

For Telegram, a static ngrok domain is strongly recommended.

## Useful Commands

```bash
docker compose up --build
docker compose up -d --build
docker compose ps
docker compose logs -f api
docker compose logs -f ngrok
docker compose logs telegram-webhook
docker compose stop
docker compose start
docker compose down
docker compose build --no-cache api
docker builder prune -af
docker system prune -af
```
