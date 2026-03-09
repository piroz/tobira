FROM rspamd/rspamd:latest

# Install tobira plugin
COPY integrations/rspamd/tobira.lua /etc/rspamd/plugins.d/tobira.lua

# Register tobira as a module (base config + local.d override)
COPY docker/rspamd/modules.d/tobira.conf /usr/share/rspamd/config/modules.d/tobira.conf
COPY docker/rspamd/local.d/ /etc/rspamd/local.d/

EXPOSE 11333 11334
