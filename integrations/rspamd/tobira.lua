--[[
  tobira rspamd plugin
  Queries the tobira API server for spam classification and inserts
  symbols based on the prediction score.

  Requires rspamd >= 3.0 (lua_http)
]]--

local lua_util = require "lua_util"
local rspamd_http = require "rspamd_http"
local rspamd_logger = require "rspamd_logger"
local ucl = require "ucl"

local N = "tobira"

local LABEL_SPAM = "spam"
local LABEL_HAM = "ham"

local PREDICT_HEADERS = { ["Content-Type"] = "application/json" }

-- Symbol tiers ordered by threshold descending for score matching
local SYMBOL_TIERS = {
  { key = "spam_high", op = ">=" },
  { key = "spam_med",  op = ">=" },
  { key = "spam_low",  op = ">=" },
  { key = "ham",       op = "<"  },
}

local default_config = {
  api_url = "http://127.0.0.1:8000",
  timeout = 2.0,
  max_text_bytes = 65536,
  fail_open = true,
  symbols = {
    spam_high = {
      name = "TOBIRA_SPAM_HIGH",
      weight = 8.0,
      threshold = 0.9,
    },
    spam_med = {
      name = "TOBIRA_SPAM_MED",
      weight = 5.0,
      threshold = 0.7,
    },
    spam_low = {
      name = "TOBIRA_SPAM_LOW",
      weight = 2.0,
      threshold = 0.5,
    },
    ham = {
      name = "TOBIRA_HAM",
      weight = -3.0,
      threshold = 0.3,
    },
    fail = {
      name = "TOBIRA_FAIL",
      weight = 0.0,
    },
  },
}

local config = {}
local predict_url = ""

local function parse_response(body)
  local parser = ucl.parser()
  local ok, err = parser:parse_string(body)
  if not ok then
    return nil, err
  end
  return parser:get_object(), nil
end

local function encode_json(tbl)
  return ucl.to_format(tbl, "json-compact")
end

local function get_spam_score(result)
  if result.labels and result.labels[LABEL_SPAM] then
    return result.labels[LABEL_SPAM]
  end
  if result.label == LABEL_SPAM then
    return result.score
  end
  if result.label == LABEL_HAM then
    return 1.0 - result.score
  end
  return nil
end

local function insert_symbol(task, spam_score)
  local syms = config.symbols
  local score_opt = string.format("score=%.4f", spam_score)

  for _, tier in ipairs(SYMBOL_TIERS) do
    local sym = syms[tier.key]
    local match = (tier.op == ">=" and spam_score >= sym.threshold)
                  or (tier.op == "<" and spam_score < sym.threshold)
    if match then
      task:insert_result(sym.name, 1.0, score_opt)
      return
    end
  end
end

local function fail_result(task, reason, fmt, ...)
  rspamd_logger.warnx(task, fmt, ...)
  if config.fail_open then
    task:insert_result(config.symbols.fail.name, 1.0, reason)
  end
end

local function tobira_callback(task)
  local parts = task:get_text_parts()
  if not parts or #parts == 0 then
    return
  end

  local text_content = {}
  for _, part in ipairs(parts) do
    local content = part:get_content()
    if content then
      table.insert(text_content, tostring(content))
    end
  end

  local text = table.concat(text_content, "\n")
  if #text == 0 then
    return
  end

  if #text > config.max_text_bytes then
    text = string.sub(text, 1, config.max_text_bytes)
  end

  local request_body = encode_json({ text = text })

  local function http_callback(err, code, body)
    if err then
      fail_result(task, "http_error",
        "%s: HTTP request failed: %s", N, err)
      return
    end

    if code ~= 200 then
      fail_result(task, string.format("http_%s", code),
        "%s: API returned HTTP %s", N, code)
      return
    end

    local result, parse_err = parse_response(body)
    if not result then
      fail_result(task, "parse_error",
        "%s: failed to parse response: %s", N, parse_err)
      return
    end

    local spam_score = get_spam_score(result)
    if not spam_score then
      fail_result(task, "no_score",
        "%s: could not determine spam score", N)
      return
    end

    rspamd_logger.infox(task, "%s: label=%s score=%.4f spam_score=%.4f",
      N, result.label, result.score, spam_score)
    insert_symbol(task, spam_score)
  end

  rspamd_http.request({
    task = task,
    url = predict_url,
    body = request_body,
    headers = PREDICT_HEADERS,
    timeout = config.timeout,
    callback = http_callback,
  })
end

local function configure(cfg)
  local opts = cfg:get_all_opt(N)
  if not opts then
    rspamd_logger.warnx(rspamd_config, "%s: no configuration found", N)
    return false
  end

  config = lua_util.override_defaults(default_config, opts)
  predict_url = config.api_url .. "/predict"
  return true
end

if not configure(rspamd_config) then
  rspamd_logger.errx(rspamd_config, "%s: plugin disabled: bad config", N)
  return
end

local id = rspamd_config:register_symbol({
  name = config.symbols.spam_high.name,
  type = "normal",
  callback = tobira_callback,
  score = config.symbols.spam_high.weight,
})

for _, key in ipairs({"spam_med", "spam_low", "ham", "fail"}) do
  rspamd_config:register_symbol({
    name = config.symbols[key].name,
    type = "virtual",
    parent = id,
    score = config.symbols[key].weight,
  })
end

rspamd_logger.infox(rspamd_config, "%s: plugin loaded, api_url=%s",
  N, config.api_url)
