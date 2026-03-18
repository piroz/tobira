# SpamAssassin Integration

This guide covers integrating tobira with SpamAssassin using a custom Perl plugin.

## Prerequisites

- SpamAssassin >= 3.4
- Perl module `LWP::UserAgent` (usually pre-installed)
- Perl module `JSON` (`sudo cpan JSON` or `sudo apt install libjson-perl`)
- tobira API server running and accessible

## Installation

1. Copy the plugin file:

    ```bash
    sudo cp integrations/spamassassin/Tobira.pm /etc/spamassassin/
    ```

2. Copy the rule configuration:

    ```bash
    sudo cp integrations/spamassassin/tobira.cf /etc/spamassassin/
    ```

3. Edit `/etc/spamassassin/tobira.cf` to adjust settings:

    ```perl
    loadplugin Mail::SpamAssassin::Plugin::Tobira tobira.pm

    tobira_url      http://127.0.0.1:8000/predict
    tobira_timeout  5
    tobira_send_headers  0

    # Rules and scores
    header   TOBIRA_SPAM_HIGH  eval:check_tobira()
    score    TOBIRA_SPAM_HIGH  8.0
    describe TOBIRA_SPAM_HIGH  tobira: high confidence spam (>= 0.9)

    header   TOBIRA_SPAM_MED   eval:check_tobira()
    score    TOBIRA_SPAM_MED   5.0
    describe TOBIRA_SPAM_MED   tobira: medium confidence spam (>= 0.7)

    header   TOBIRA_SPAM_LOW   eval:check_tobira()
    score    TOBIRA_SPAM_LOW   2.0
    describe TOBIRA_SPAM_LOW   tobira: low confidence spam (>= 0.5)

    header   TOBIRA_HAM        eval:check_tobira()
    score    TOBIRA_HAM        -3.0
    describe TOBIRA_HAM        tobira: likely legitimate (< 0.3)

    header   TOBIRA_FAIL       eval:check_tobira()
    score    TOBIRA_FAIL       0.0
    describe TOBIRA_FAIL       tobira: API request failed
    ```

4. Restart SpamAssassin:

    ```bash
    sudo systemctl restart spamassassin
    ```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `tobira_url` | `http://127.0.0.1:8000/predict` | tobira API endpoint URL |
| `tobira_timeout` | `5` | HTTP request timeout in seconds |
| `tobira_send_headers` | `0` | Send email headers (0 = disabled, 1 = enabled) |

## Header Analysis

Set `tobira_send_headers 1` to include SPF, DKIM, DMARC results and sender information in the request.

## Tags

The plugin sets the following tags on each message:

| Tag | Description |
|-----|-------------|
| `TOBIRALABEL` | Classification label (`spam` or `ham`) |
| `TOBIRASCORE` | ML spam probability (0.0 to 1.0) |

Access these in custom rules or templates:

```perl
# In a template
_TOBIRALABEL_ scored _TOBIRASCORE_
```

## Verification

Run SpamAssassin in lint mode to check the plugin loads correctly:

```bash
spamassassin --lint 2>&1 | grep -i tobira
```

Test with a sample message:

```bash
echo "Subject: Buy cheap watches" | spamassassin -t 2>&1 | grep TOBIRA
```

## Troubleshooting

**Plugin not loading**: Check that `Tobira.pm` is in the SpamAssassin plugin directory and `tobira.cf` is in the rules directory.

**JSON module missing**: Install with `sudo cpan JSON` or your distribution's package manager.

**Timeout errors**: Increase `tobira_timeout` or verify API server connectivity.
