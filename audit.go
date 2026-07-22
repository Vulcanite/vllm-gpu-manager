package main

import (
	"bufio"
	"encoding/json"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

type auditEvent struct {
	Timestamp     time.Time              `json:"timestamp"`
	Action        string                 `json:"action"`
	Result        string                 `json:"result"`
	Actor         string                 `json:"actor"`
	Summary       string                 `json:"summary"`
	Model         string                 `json:"model,omitempty"`
	Port          int                    `json:"port,omitempty"`
	ContainerID   string                 `json:"container_id,omitempty"`
	ContainerName string                 `json:"container_name,omitempty"`
	Username      string                 `json:"username,omitempty"`
	Params        map[string]interface{} `json:"params,omitempty"`
	Error         string                 `json:"error,omitempty"`
}

func appendAuditEvent(event auditEvent) {
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now().UTC()
	}
	event.Action = strings.TrimSpace(event.Action)
	event.Result = strings.TrimSpace(event.Result)
	if event.Actor == "" {
		event.Actor = "dashboard"
	}
	event.Params = sanitizeAuditParams(event.Params)

	data, err := json.Marshal(event)
	if err != nil {
		log.Printf("[AUDIT] failed to marshal event %q: %v", event.Action, err)
		return
	}
	path := auditLogFilePath()
	if err := ensureRuntimeDir(path, 0o750); err != nil {
		log.Printf("[AUDIT] failed to prepare audit directory: %v", err)
		return
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		log.Printf("[AUDIT] failed to open %s: %v", path, err)
		return
	}
	defer file.Close()
	if _, err := file.Write(append(data, '\n')); err != nil {
		log.Printf("[AUDIT] failed to write %s: %v", path, err)
		return
	}
	log.Printf("[AUDIT] action=%s result=%s actor=%s summary=%q", event.Action, event.Result, event.Actor, event.Summary)
}

func readAuditEvents(limit int) ([]auditEvent, error) {
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	file, err := os.Open(auditLogFilePath())
	if err != nil {
		if os.IsNotExist(err) {
			return []auditEvent{}, nil
		}
		return nil, err
	}
	defer file.Close()

	var events []auditEvent
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var event auditEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}
		events = append(events, event)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(events) > limit {
		events = events[len(events)-limit:]
	}
	for i, j := 0, len(events)-1; i < j; i, j = i+1, j-1 {
		events[i], events[j] = events[j], events[i]
	}
	return events, nil
}

func sanitizeAuditParams(params map[string]interface{}) map[string]interface{} {
	if len(params) == 0 {
		return nil
	}
	cleaned := make(map[string]interface{}, len(params))
	for key, value := range params {
		if isSecretAuditParamKey(key) {
			continue
		}
		cleaned[key] = value
	}
	if len(cleaned) == 0 {
		return nil
	}
	return cleaned
}

func isSecretAuditParamKey(key string) bool {
	return strings.Contains(key, "api_key") ||
		key == "token" ||
		strings.HasSuffix(key, "_token") ||
		strings.Contains(key, "secret") ||
		strings.Contains(key, "password")
}

func auditLaunchParams(req startModelRequest, exposure string, bindIP string) map[string]interface{} {
	data, err := json.Marshal(req)
	if err != nil {
		return map[string]interface{}{}
	}
	params := map[string]interface{}{}
	if err := json.Unmarshal(data, &params); err != nil {
		return map[string]interface{}{}
	}
	params["speculative_config_set"] = strings.TrimSpace(req.SpeculativeConfig) != ""
	params["bind_ip"] = bindIP
	params["exposure"] = exposure
	params["use_custom_server_api_key"] = strings.TrimSpace(req.ServerAPIKey) != ""
	delete(params, "speculative_config")
	return params
}

func auditedServerAPIKeyDeletes(names []string) {
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		appendAuditEvent(auditEvent{
			Action:        "server_api_key_deleted",
			Result:        "success",
			ContainerName: name,
			Summary:       "Deleted per-server API key for " + name,
		})
	}
}

func parseAuditLimit(raw string) int {
	limit, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return 200
	}
	return limit
}
