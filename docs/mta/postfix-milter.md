# Postfix milter Integration

This guide covers integrating tobira with Postfix using the Python milter daemon.

## Prerequisites

- Postfix mail server
- Python 3.9+
- `libmilter-dev` system package
- tobira API server running and accessible

## Installation

1. Install tobira with milter support:

    ```bash
    pip install tobira[milter]
    ```

    !!! note
        The `pymilter` package requires `libmilter-dev`. Install it first:

        === "Debian/Ubuntu"

            ```bash
            sudo apt install libmilter-dev
            ```

        === "RHEL/CentOS"

            ```bash
            sudo dnf install sendmail-milter-devel
            ```

2. Create the configuration file `/etc/tobira/milter.conf`:

    ```ini
    [milter]
    api_url = http://127.0.0.1:8000/predict
    socket = unix:/var/run/tobira/milter.sock
    timeout = 10
    fail_action = accept
    reject_threshold = 0.9
    add_headers = true
    ```

3. Create the socket directory:

    ```bash
    sudo mkdir -p /var/run/tobira
    sudo chown tobira:tobira /var/run/tobira
    ```

4. Install the systemd service:

    ```bash
    sudo cp integrations/postfix-milter/tobira-milter.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now tobira-milter
    ```

5. Configure Postfix to use the milter. Add to `/etc/postfix/main.cf`:

    ```
    smtpd_milters = unix:/var/run/tobira/milter.sock
    non_smtpd_milters = unix:/var/run/tobira/milter.sock
    milter_default_action = accept
    ```

6. Reload Postfix:

    ```bash
    sudo systemctl reload postfix
    ```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `api_url` | `http://127.0.0.1:8000/predict` | tobira API endpoint URL |
| `socket` | `unix:/var/run/tobira/milter.sock` | Milter socket path (UNIX or TCP) |
| `timeout` | `10` | API connection timeout in seconds |
| `fail_action` | `accept` | Action on API failure: `accept` or `tempfail` |
| `reject_threshold` | `0.9` | Score threshold for rejection (0 to disable) |
| `add_headers` | `true` | Add `X-Tobira-Score` and `X-Tobira-Label` headers |

## Socket Types

The milter supports both UNIX and TCP sockets:

```ini
# UNIX socket (recommended for same-host deployment)
socket = unix:/var/run/tobira/milter.sock

# TCP socket (for remote deployment)
socket = inet:8899@127.0.0.1
```

## Headers Added

When `add_headers = true`, the milter adds these headers to each email:

| Header | Example | Description |
|--------|---------|-------------|
| `X-Tobira-Score` | `0.95` | ML spam probability |
| `X-Tobira-Label` | `spam` | Classification label |

## Fail Handling

| `fail_action` | Behavior |
|----------------|----------|
| `accept` | Accept the email without ML scoring (fail-open) |
| `tempfail` | Return a temporary failure, asking the sender to retry |

## Systemd Service

The included systemd service file runs the milter with security hardening:

- Runs as dedicated `tobira` user
- `ProtectSystem=strict` — Read-only filesystem
- `ProtectHome=true` — No access to home directories
- `NoNewPrivileges=true` — Cannot gain additional privileges
- `PrivateTmp=true` — Isolated temporary directory

## Verification

Check the milter is running:

```bash
sudo systemctl status tobira-milter
```

Check the socket exists:

```bash
ls -la /var/run/tobira/milter.sock
```

Send a test email and verify headers:

```bash
# Check for tobira headers in received mail
grep X-Tobira /var/mail/testuser
```

## Troubleshooting

**Socket permission denied**: Ensure the `tobira` user has write access to the socket directory and Postfix can read the socket.

**pymilter installation fails**: Install `libmilter-dev` before installing the Python package.

**Emails being rejected**: Check `reject_threshold` setting. Set to `0` to disable rejection and only add headers.
