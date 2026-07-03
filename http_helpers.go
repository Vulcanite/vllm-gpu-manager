package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"vllm-manager/api"
)

type appError struct {
	Status int
	Detail string
}

func (e appError) Error() string {
	return e.Detail
}

func newAppError(status int, format string, args ...interface{}) error {
	return appError{Status: status, Detail: fmt.Sprintf(format, args...)}
}

func writeJSON(w http.ResponseWriter, value interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(value)
}

func writeJSONError(w http.ResponseWriter, status int, detail string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"detail": detail})
}

// redactSensitive scrubs known secret forms from a human-readable string
// before it is logged or returned to the caller. It is intentionally
// conservative: it only redacts the exact prefixes we control, and it
// preserves the original whitespace/newlines as much as possible.
func redactSensitive(detail string) string {
	if detail == "" {
		return detail
	}
	lines := strings.Split(detail, "\n")
	for i, line := range lines {
		lines[i] = redactSensitiveLine(line)
	}
	return strings.Join(lines, "\n")
}

func redactSensitiveLine(line string) string {
	// Replace obvious key=value/flag=value forms in place; this works whether
	// the value is quoted, contains spaces inside the quotes, or not.
	line = redactKeyValue(line, "HF_TOKEN")
	line = redactKeyValue(line, "VLLM_API_KEY")
	line = redactKeyValue(line, "VLLM_MANAGER_PASSWORD")
	line = redactKeyValue(line, "inference_api_key")
	line = redactKeyValue(line, "inference-key")
	line = redactFlag(line, "--api-key")
	line = redactFlag(line, "--inference-key")
	line = redactFlag(line, "-H", "Authorization:")
	// Bare token leakage in shell-style arguments like "hf_token=abc".
	for _, key := range []string{"hf_token"} {
		line = redactKeyValue(line, key)
	}
	return line
}

// redactKeyValue replaces any occurrence of key= or key="..." with key=<redacted>.
func redactKeyValue(line, key string) string {
	prefix := key + "="
	idx := 0
	for {
		start := strings.Index(line[idx:], prefix)
		if start < 0 {
			return line
		}
		start += idx
		end := start + len(prefix)
		if end < len(line) && line[end] == '"' {
			closeIdx := strings.Index(line[end+1:], "\"")
			if closeIdx < 0 {
				line = line[:end+1] + "<redacted>"
				return line
			}
			line = line[:end+1] + "<redacted>" + line[end+1+closeIdx:]
			idx = end + 1 + len("<redacted>") + closeIdx
		} else {
			// Value runs to the next whitespace or end of line.
			j := end
			for j < len(line) && line[j] != ' ' && line[j] != '\t' {
				j++
			}
			line = line[:end] + "<redacted>" + line[j:]
			idx = end + len("<redacted>")
		}
	}
}

// redactFlag replaces the value following a flag (either "--api-key VALUE" or
// "--api-key=VALUE" if the second style is requested). The optional hint is
// used to also redact the value following an "-H" header argument when the
// next token begins with the hint.
func redactFlag(line, flag string, hint ...string) string {
	prefix := flag + "="
	idx := 0
	for {
		start := strings.Index(line[idx:], prefix)
		if start < 0 {
			break
		}
		start += idx
		end := start + len(prefix)
		j := end
		for j < len(line) && line[j] != ' ' && line[j] != '\t' {
			j++
		}
		line = line[:end] + "<redacted>" + line[j:]
		idx = end + len("<redacted>")
	}

	// Now handle the space-separated form. We work on the original line so
	// the original whitespace and quoting (which is meaningful for -H values)
	// is preserved as much as possible.
	if flag == "--api-key" || flag == "--inference-key" {
		return redactSpaceSeparatedFlag(line, flag)
	}
	if len(hint) > 0 && hint[0] != "" {
		return redactHeaderLine(line, hint[0])
	}
	return line
}

// redactSpaceSeparatedFlag replaces the whitespace-delimited token that
// immediately follows the named flag with "<redacted>". It tolerates the flag
// appearing inside a longer word (e.g. "--api-key-extra") by requiring a
// word boundary on the right.
func redactSpaceSeparatedFlag(line, flag string) string {
	idx := 0
	for {
		start := strings.Index(line[idx:], flag)
		if start < 0 {
			return line
		}
		start += idx
		// Make sure we are matching the whole flag, not a prefix.
		end := start + len(flag)
		if start > 0 {
			prev := line[start-1]
			if prev != ' ' && prev != '\t' {
				idx = end
				continue
			}
		}
		if end < len(line) {
			next := line[end]
			if next != ' ' && next != '\t' {
				idx = end
				continue
			}
		}
		// Skip the flag itself and any whitespace.
		j := end
		for j < len(line) && (line[j] == ' ' || line[j] == '\t') {
			j++
		}
		// Now j points at the start of the value. Eat until the next
		// whitespace. We deliberately stop at the first whitespace so we
		// don't accidentally absorb the next argument.
		k := j
		for k < len(line) && line[k] != ' ' && line[k] != '\t' {
			k++
		}
		return line[:j] + "<redacted>" + line[k:]
	}
}

// redactHeaderLine redacts the value of a "-H" header whose name starts with
// the given hint. It preserves the original spacing and quoting and replaces
// only the value portion after the hint's "Name: " separator.
func redactHeaderLine(line, hintPrefix string) string {
	idx := 0
	for {
		start := strings.Index(line[idx:], "-H ")
		if start < 0 {
			if start = strings.Index(line[idx:], "-H\t"); start < 0 {
				return line
			}
		}
		start += idx
		// Walk past the "-H" and any whitespace.
		j := start + 2
		for j < len(line) && (line[j] == ' ' || line[j] == '\t') {
			j++
		}
		// j now points at the start of the value. Tolerate a leading
		// double-quote by adjusting j past it; we'll later rebuild the
		// line by dropping that quote so the placeholder is unquoted.
		hadLeadingQuote := false
		if j < len(line) && line[j] == '"' {
			hadLeadingQuote = true
			j++
		}
		if j+len(hintPrefix) > len(line) || line[j:j+len(hintPrefix)] != hintPrefix {
			idx = j
			continue
		}
		// valueStart points at the start of the value (the byte after
		// the hint). We want to replace the value with "<redacted>" and
		// keep the colon + space that separates the hint from the value.
		valueStart := j + len(hintPrefix)
		// Skip a single separator space if present so the replacement
		// is followed by whitespace rather than a glued-on space.
		separatorEnd := valueStart
		if separatorEnd < len(line) && line[separatorEnd] == ' ' {
			separatorEnd++
		}
		// Find the end of the value: closing quote if quoted, else next
		// whitespace. For a quoted value we also include the closing
		// quote in the dropped portion so the rebuild is clean.
		valueEnd := separatorEnd
		if hadLeadingQuote {
			closeIdx := strings.Index(line[valueEnd:], `"`)
			if closeIdx < 0 {
				valueEnd = len(line)
			} else {
				valueEnd = valueEnd + closeIdx + 1
			}
		} else {
			for valueEnd < len(line) && line[valueEnd] != ' ' && line[valueEnd] != '\t' {
				valueEnd++
			}
		}
		// Rebuild the line, preserving the leading quote if it was there
		// (we still emit it) and replacing the value with a placeholder.
		prefixEnd := j
		if hadLeadingQuote {
			prefixEnd = j - 1
		}
		return line[:prefixEnd] + hintPrefix + " <redacted>" + line[valueEnd:]
	}
}

func writeAPIError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	if api.IsValidationError(err) {
		status = http.StatusBadRequest
	}
	var appErr appError
	if errors.As(err, &appErr) {
		status = appErr.Status
	}
	log.Printf("[API] error: %s -> %d", redactSensitive(err.Error()), status)
	writeJSONError(w, status, err.Error())
}

func decodeJSONBody(w http.ResponseWriter, r *http.Request, value interface{}) bool {
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return false
	}
	return true
}

func isAllowedOrigin(origin string, requestHost string) bool {
	parsed, err := url.Parse(origin)
	if err != nil || parsed.Host == "" {
		return false
	}

	if parsed.Host == requestHost {
		return true
	}

	host := parsed.Hostname()
	if host == "localhost" || host == "127.0.0.1" || host == "::1" {
		return true
	}

	for _, allowed := range strings.Split(os.Getenv("VLLM_MANAGER_ALLOWED_ORIGINS"), ",") {
		if strings.TrimSpace(allowed) == origin {
			return true
		}
	}
	return false
}

type responseRecorder struct {
	http.ResponseWriter
	status int
}

func (r *responseRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func (r *responseRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		recorder := &responseRecorder{ResponseWriter: w, status: http.StatusOK}
		defer func() {
			log.Printf("[API] %s %s -> %d (%s)", r.Method, r.URL.Path, recorder.status, time.Since(start).Round(time.Millisecond))
		}()

		origin := r.Header.Get("Origin")
		if origin != "" {
			if !isAllowedOrigin(origin, r.Host) {
				writeJSONError(recorder, http.StatusForbidden, "origin not allowed")
				return
			}
			recorder.Header().Set("Access-Control-Allow-Origin", origin)
			recorder.Header().Set("Vary", "Origin")
			recorder.Header().Set("Access-Control-Allow-Credentials", "true")
		}
		recorder.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		recorder.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding")
		if r.Method == "OPTIONS" {
			recorder.WriteHeader(http.StatusOK)
			return
		}
		next(recorder, r)
	}
}
