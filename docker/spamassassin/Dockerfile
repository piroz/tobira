FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      spamassassin \
      spamd \
      spamc \
      libwww-perl \
      libjson-perl \
      perl \
      procmail \
    && rm -rf /var/lib/apt/lists/*

# Install tobira plugin
COPY integrations/spamassassin/Tobira.pm /etc/spamassassin/Tobira.pm
COPY integrations/spamassassin/tobira.cf /etc/spamassassin/tobira.cf

# Override tobira_url to point to Docker service
# File named v99_* so it loads after tobira.cf (alphabetical order)
COPY docker/spamassassin/v99_tobira_docker.cf /etc/spamassassin/v99_tobira_docker.cf

EXPOSE 783

# Debian bookworm: spamd is at /usr/sbin/spamd
CMD ["/usr/sbin/spamd", "--listen=0.0.0.0:783", "--allow-tell", "--nouser-config", "--local", "-D"]
