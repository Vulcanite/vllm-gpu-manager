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

func redactSensitive(detail string) string {
	detail = strings.Replace(detail, "hf_token", "<redacted>", -1)
	parts := strings.Fields(detail)
	for i, p := range parts {
		if strings.HasPrefix(p, "HF_TOKEN=") {
			parts[i] = "HF_TOKEN=<redacted>"
		}
		if (p == "--api-key" || (p == "-H" && i+1 < len(parts) && strings.HasPrefix(parts[i+1], "Authorization:"))) && i+1 < len(parts) {
			parts[i+1] = "<redacted>"
		}
	}
	return strings.Join(parts, " ")
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
